#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <string>
#include <memory>
#include <csignal>
#include <atomic>
#include <cmath>
#include <iomanip> // For std::fixed and std::setprecision
#include <future>  // For std::promise, std::future

// ALSA, RTC, and now CURL headers
#include <alsa/asoundlib.h>
#include <curl/curl.h>
#include "bytertc_video.h"
#include "bytertc_room.h"
#include "bytertc_video_event_handler.h"
#include "bytertc_room_event_handler.h"
#include "rtc/bytertc_defines.h"
#include "rtc/bytertc_audio_defines.h" // **修正**: 添加 #include 关键字
#include "rtc/bytertc_media_defines.h"
#include "rtc/bytertc_audio_frame.h"
#include "rtc/bytertc_audio_device_manager.h"
#include "util/json11/json11.hpp"

// ======================= 全局变量和状态机 =======================
std::atomic<bool> g_should_exit(false);
enum class AppState { WAITING_FOR_TRIGGER, CREATING_ROOM, IN_CONVERSATION, COOLDOWN };
std::atomic<AppState> g_app_state(AppState::WAITING_FOR_TRIGGER);

// **修正**: 用于RTC线程向主线程发送日志和消息。
std::shared_ptr<std::promise<std::string>> g_log_promise_ptr_main;
std::shared_ptr<std::future<std::string>> g_log_future_ptr_main;

// **修正**: 改进的 send_log_to_main 函数
void send_log_to_main(const std::string& log_message) {
    // 检查 promise 是否已初始化，并且其对应的 future 是否尚未就绪（即可以写入新值）
    if (g_log_promise_ptr_main &&
        g_log_future_ptr_main && // 确保 future 也已初始化
        g_log_future_ptr_main->wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
        try {
            g_log_promise_ptr_main->set_value(log_message);
        } catch (const std::future_error& e) {
            std::cerr << "[Thread Log Error - Promise Already Set or Invalid]: " << log_message << " (" << e.what() << ")" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[Thread Log Error - General Exception]: " << log_message << " (" << e.what() << ")" << std::endl;
        }
    } else {
        // 如果 promise 未初始化、或 future 已就绪、或 g_log_future_ptr_main 为空，则直接打印
        std::cerr << "[Direct Log (No Active Promise/Promise Already Set)]: " << log_message << std::endl;
    }
}


void signal_handler(int signum) {
    if (signum == SIGINT) {
        std::cout << "\nCtrl+C pressed. Shutting down..." << std::endl;
        g_should_exit = true;
    }
}

// ======================= HTTP 请求相关 =======================
struct CozeRoomInfo {
    std::string app_id;
    std::string token;
    std::string room_id;
    std::string user_id;
    std::string bot_id = "7523153892465688627"; // Hardcode your bot_id here
};

static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

// **修正**: create_coze_room_request 返回 bool 并通过引用传递结果
bool create_coze_room_request(const std::string& pat, const std::string& bot_id, CozeRoomInfo& out_room_info) {
    CURL *curl;
    CURLcode res;
    std::string readBuffer;

    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();
    if(curl) {
        struct curl_slist *headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        std::string auth_header = "Authorization: Bearer " + pat;
        headers = curl_slist_append(headers, auth_header.c_str());

        std::string post_data = "{\"bot_id\": \"" + bot_id + "\"}";

        curl_easy_setopt(curl, CURLOPT_URL, "https://api.coze.cn/v1/audio/rooms");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, post_data.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);
        // **修正**: 临时禁用SSL验证和Name Resolution Cache
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L); // 仅用于测试！生产环境严禁禁用！
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);   // 仅用于测试！生产环境严禁禁用！
        curl_easy_setopt(curl, CURLOPT_DNS_CACHE_TIMEOUT, 0L); // 禁用DNS缓存
        curl_easy_setopt(curl, CURLOPT_FORBID_REUSE, 1L);     // 禁用连接重用，强制每次都重新解析

        res = curl_easy_perform(curl);
        if(res != CURLE_OK) {
            send_log_to_main("curl_easy_perform() failed: " + std::string(curl_easy_strerror(res)));
            curl_slist_free_all(headers);
            curl_easy_cleanup(curl);
            curl_global_cleanup();
            return false;
        }

        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    }
    curl_global_cleanup();

    send_log_to_main("Coze API response: " + readBuffer);

    std::string err;
    json11::Json json = json11::Json::parse(readBuffer, err);
    if (!err.empty() || json["code"].int_value() != 0) {
        send_log_to_main("Failed to parse Coze API response or API returned error: " + json["msg"].string_value());
        return false;
    }

    out_room_info.app_id = json["data"]["app_id"].string_value();
    out_room_info.token = json["data"]["token"].string_value();
    out_room_info.room_id = json["data"]["room_id"].string_value();
    out_room_info.user_id = json["data"]["uid"].string_value();
    out_room_info.bot_id = bot_id;

    return true;
}

// ======================= 音频处理相关 =======================
double calculate_rms(const int16_t* pcm_data, size_t sample_count) {
    if (sample_count == 0) return 0.0;
    double sum_sq = 0.0;
    for (size_t i = 0; i < sample_count; ++i) {
        double sample = static_cast<double>(pcm_data[i]);
        sum_sq += sample * sample;
    }
    return std::sqrt(sum_sq / sample_count);
}

class AlsaAudioManager {
private:
    snd_pcm_t* play_handle_ = nullptr;
    snd_pcm_t* capture_handle_ = nullptr;
    
    std::string playback_device_name_;
    unsigned int playback_sample_rate_;
    unsigned int playback_channels_;

    std::string capture_device_name_;
    unsigned int capture_sample_rate_;
    unsigned int capture_channels_;

public:
    AlsaAudioManager(const std::string& pb_dev, unsigned int pb_rate, unsigned int pb_ch,
                     const std::string& cap_dev, unsigned int cap_rate, unsigned int cap_ch)
        : playback_device_name_(pb_dev), playback_sample_rate_(pb_rate), playback_channels_(pb_ch),
          capture_device_name_(cap_dev), capture_sample_rate_(cap_rate), capture_channels_(cap_ch) {}

    bool init() {
        if (!initPlayback()) return false;
        if (!initCapture()) return false;
        return true;
    }
    
    int read(int16_t* buffer, int samples) {
        if (!capture_handle_) return -1;
        return snd_pcm_readi(capture_handle_, buffer, samples);
    }
    
    void play(const uint8_t* pcm_data, size_t data_size) {
        if (!play_handle_) return;
        if (playback_channels_ == 0) return;
        snd_pcm_uframes_t frames_to_write = data_size / (playback_channels_ * sizeof(int16_t));
        if (frames_to_write == 0) return;
        
        int rc = snd_pcm_writei(play_handle_, pcm_data, frames_to_write);
        if (rc == -EPIPE) {
            snd_pcm_prepare(play_handle_);
        } else if (rc < 0) {
            send_log_to_main("Error playing audio: " + std::string(snd_strerror(rc)));
        }
    }

    void release() {
        if (play_handle_) {
            send_log_to_main("Closing playback device: " + playback_device_name_);
            snd_pcm_close(play_handle_);
            play_handle_ = nullptr;
        }
        if (capture_handle_) {
            send_log_to_main("Closing capture device: " + capture_device_name_);
            snd_pcm_close(capture_handle_);
            capture_handle_ = nullptr;
        }
    }

private:
    bool initPlayback() {
        int rc;
        rc = snd_pcm_open(&play_handle_, playback_device_name_.c_str(), SND_PCM_STREAM_PLAYBACK, 0);
        if (rc < 0) { send_log_to_main("Playback open error: " + std::string(snd_strerror(rc))); return false; }
        
        snd_pcm_hw_params_t* params;
        snd_pcm_hw_params_alloca(&params); // **修正**: 将 ¶ms 改为 ¶ms
        
        snd_pcm_hw_params_any(play_handle_, params);
        snd_pcm_hw_params_set_access(play_handle_, params, SND_PCM_ACCESS_RW_INTERLEAVED);
        snd_pcm_hw_params_set_format(play_handle_, params, SND_PCM_FORMAT_S16_LE);
        snd_pcm_hw_params_set_channels(play_handle_, params, playback_channels_);
        snd_pcm_hw_params_set_rate_near(play_handle_, params, &playback_sample_rate_, nullptr);
        
        rc = snd_pcm_hw_params(play_handle_, params);
        if (rc < 0) { send_log_to_main("Playback set H/W params error: " + std::string(snd_strerror(rc))); return false; }
        
        send_log_to_main("Playback device initialized: " + playback_device_name_);
        return true;
    }

    bool initCapture() {
        int rc;
        rc = snd_pcm_open(&capture_handle_, capture_device_name_.c_str(), SND_PCM_STREAM_CAPTURE, 0);
        if (rc < 0) { send_log_to_main("Capture open error: " + std::string(snd_strerror(rc))); return false; }
        
        snd_pcm_hw_params_t* params;
        snd_pcm_hw_params_alloca(&params); // **修正**: 将 ¶ms 改为 ¶ms
        
        snd_pcm_hw_params_any(capture_handle_, params);
        snd_pcm_hw_params_set_access(capture_handle_, params, SND_PCM_ACCESS_RW_INTERLEAVED);
        snd_pcm_hw_params_set_format(capture_handle_, params, SND_PCM_FORMAT_S16_LE);
        snd_pcm_hw_params_set_channels(capture_handle_, params, capture_channels_);
        snd_pcm_hw_params_set_rate_near(capture_handle_, params, &capture_sample_rate_, nullptr);
        
        rc = snd_pcm_hw_params(capture_handle_, params);
        if (rc < 0) { send_log_to_main("Capture set H/W params error: " + std::string(snd_strerror(rc))); return false; }
        
        send_log_to_main("Capture device initialized: " + capture_device_name_);
        return true;
    }
};

class MyRTCEventHandler : public bytertc::IRTCVideoEventHandler,
                          public bytertc::IRTCRoomEventHandler,
                          public bytertc::IAudioFrameObserver {
public:
    bytertc::IRTCRoom* rtc_room_ = nullptr;
    AlsaAudioManager* alsa_manager_ = nullptr;
    std::string bot_id_;

    MyRTCEventHandler(AlsaAudioManager* manager, std::string bot_id) 
        : alsa_manager_(manager), bot_id_(bot_id) {}
    ~MyRTCEventHandler() override = default;

    void onWarning(int warn) override { send_log_to_main("[RTC Video Warning] Code: " + std::to_string(warn)); }
    void onError(int err) override { send_log_to_main("[RTC Video Error] Code: " + std::to_string(err)); }
    
    void onRoomStateChanged(const char* room_id, const char* uid, int state, const char* extra_info) override {
        send_log_to_main("[Room State Changed] room_id: " + std::string(room_id) + ", uid: " + std::string(uid) 
                  + ", state: " + std::to_string(state) + ", extra_info: " + std::string(extra_info));
        if (state == 0) { 
            send_log_to_main("[Room State] Joined room successfully!");
            // **关键**：加入房间成功后，发送一个事件告诉Bot我们准备好了
            if (rtc_room_ && !bot_id_.empty()) {
                json11::Json session_update = json11::Json::object {
                    { "id", "client_event_session_update" }, 
                    { "event_type", "session.update" },
                    { "data", json11::Json::object {
                        { "event_subscriptions", json11::Json::array {}}
                    }}
                };
                std::string msg = session_update.dump();
                // **关键**：使用 sendUserMessage，因为文档说的是 UserMessage
                rtc_room_->sendUserMessage(bot_id_.c_str(), msg.c_str());
                send_log_to_main("Sent session.update to bot: " + bot_id_);
            } else {
                send_log_to_main("Warning: rtc_room_ is null or bot_id is empty, cannot send session.update.");
            }
        }
    }
    void onLeaveRoom(const bytertc::RtcRoomStats& stats) override {
        send_log_to_main("[Room Event] Left room.");
        g_app_state = AppState::COOLDOWN;
    }
    void onUserJoined(const bytertc::UserInfo& user_info, int elapsed) override {
        send_log_to_main("[Room Event] User joined! User ID: " + std::string(user_info.uid));
        if (rtc_room_) {
            rtc_room_->subscribeStream(user_info.uid, bytertc::kMediaStreamTypeAudio);
            send_log_to_main("Subscribed to User's audio stream: " + std::string(user_info.uid));
        }
    }
    void onUserLeave(const char* uid, bytertc::UserOfflineReason reason) override {
        send_log_to_main("[Room Event] User left! User ID: " + std::string(uid) + ", Reason: " + std::to_string(reason));
        if (std::string(uid) == bot_id_) { // 检查是否是Bot离开
            g_app_state = AppState::COOLDOWN;
            send_log_to_main("Bot left the room. Ending conversation.");
        }
    }
    void onUserPublishStream(const char* uid, bytertc::MediaStreamType type) override {
        if (rtc_room_ && type == bytertc::kMediaStreamTypeAudio) {
            rtc_room_->subscribeStream(uid, bytertc::kMediaStreamTypeAudio);
        }
    }
    void onUserUnpublishStream(const char* uid, bytertc::MediaStreamType type, bytertc::StreamRemoveReason reason) override {}
    void onStreamSubscribed(bytertc::SubscribeState state_code, const char* user_id, const bytertc::SubscribeConfig& info) override {}
    
    void onRoomMessageReceived(const char* uid, const char* message) override {
        send_log_to_main("[Raw Message from " + std::string(uid) + "]: " + std::string(message));
        std::string err;
        json11::Json json = json11::Json::parse(message, err);
        if (err.empty()) {
            std::string event_type = json["event_type"].string_value();
            if (event_type == "conversation.created") {
                std::string prologue = json["data"]["prologue"].string_value();
                if (!prologue.empty()) {
                    send_log_to_main("====== Bot Prologue ====== \n" + prologue + "\n==========================");
                }
            } else if (event_type == "conversation.message.completed") {
                if (json["data"]["type"].string_value() == "answer") {
                    std::string answer = json["data"]["content"].string_value();
                    send_log_to_main("====== Bot Answer ====== \n" + answer + "\n========================");
                } else if (json["data"]["type"].string_value() == "verbose") {
                    std::string verbose_content = json["data"]["content"].string_value();
                    json11::Json verbose_json = json11::Json::parse(verbose_content, err);
                    if (err.empty() && verbose_json["msg_type"].string_value() == "speech_recognize") {
                        std::string text = verbose_json["data"]["text"].string_value();
                        std::string status = verbose_json["data"]["status"].string_value();
                        send_log_to_main("[ASR Subtitle " + status + "]: " + text);
                    }
                }
            } else if (event_type == "session.updated") {
                 send_log_to_main("[Session Updated Success]");
            } else if (event_type == "error") {
                 send_log_to_main("[Bot Error Received]: " + std::to_string(json["data"]["code"].int_value()) + " - " + json["data"]["msg"].string_value());
            }
        } else {
            send_log_to_main("JSON parse error for message: " + err);
        }
    }
    
    void onRecordAudioFrameOriginal(const bytertc::IAudioFrame& audio_frame) override {}
    void onRecordAudioFrame(const bytertc::IAudioFrame& audio_frame) override {}
    void onPlaybackAudioFrame(const bytertc::IAudioFrame& audio_frame) override {}
    void onMixedAudioFrame(const bytertc::IAudioFrame& audio_frame) override {}
    void onRecordScreenAudioFrame(const bytertc::IAudioFrame& audio_frame) override {}
    void onRemoteUserAudioFrame(const bytertc::RemoteStreamKey& stream_info, const bytertc::IAudioFrame& audio_frame) override {
        if (alsa_manager_ && audio_frame.frameType() == bytertc::kAudioFrameTypePCM16) {
            alsa_manager_->play(audio_frame.data(), audio_frame.dataSize());
        }
    }
};

// ======================= 主函数 (状态机逻辑) =======================
int main() {
    signal(SIGINT, signal_handler);

    std::unique_ptr<AlsaAudioManager> alsa_manager = std::make_unique<AlsaAudioManager>(
        "hw:1,0", 48000, 1, // Playback
        "hw:0,0", 48000, 1  // Capture
    );

    if (!alsa_manager->init()) {
        std::cerr << "Failed to initialize ALSA. Exiting." << std::endl; // Main thread direct output
        return -1;
    }

    std::cout << "Application started. Listening for loud sound to trigger..." << std::endl;
    const int SAMPLES_PER_CHUNK = 480; // 48kHz, 10ms
    std::vector<int16_t> audio_buffer(SAMPLES_PER_CHUNK);
    const double VOLUME_THRESHOLD = 2000.0; // 触发对话的音量阈值

    while (!g_should_exit) {
        // **修正**: 在主循环开始处检查并打印所有来自子线程的日志
        // 在尝试获取之前确保 g_log_future_ptr_main 不为空
        if (g_log_future_ptr_main && g_log_future_ptr_main->wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
            std::cout << "\n" << g_log_future_ptr_main->get() << std::endl;
            // **修正**: 重新创建promise/future pair
            g_log_promise_ptr_main = std::make_shared<std::promise<std::string>>();
            g_log_future_ptr_main = std::make_shared<std::future<std::string>>(g_log_promise_ptr_main->get_future());
        }

        if (g_app_state == AppState::WAITING_FOR_TRIGGER) {
            int read_samples = alsa_manager->read(audio_buffer.data(), SAMPLES_PER_CHUNK);
            if (read_samples > 0) {
                double volume = calculate_rms(audio_buffer.data(), read_samples);
                // 实时打印音量，不带换行符
                std::cout << "\rVolume: " << std::fixed << std::setprecision(0) << volume << "      " << std::flush;
                
                if (volume > VOLUME_THRESHOLD) {
                    std::cout << "\nLoud sound detected! Triggering conversation..." << std::endl;
                    g_app_state = AppState::CREATING_ROOM;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        } else if (g_app_state == AppState::CREATING_ROOM) {
            CozeRoomInfo room_info;
            room_info.bot_id = "7523153892465688627"; // 你的Bot ID

            // **修正**: 重新初始化promise/future pair用于当前RTC线程的日志
            g_log_promise_ptr_main = std::make_shared<std::promise<std::string>>();
            g_log_future_ptr_main = std::make_shared<std::future<std::string>>(g_log_promise_ptr_main->get_future());

            // **修正**: Lambda 捕获列表，alsa_manager 引用捕获，room_info 值捕获
            std::thread create_room_and_rtc_thread([&alsa_manager, room_info](){ 
                CozeRoomInfo thread_room_info; // 线程内部的CozeRoomInfo，用于接收create_coze_room_request的结果
                
                // **修正**: 调用正确函数并传递结果引用
                if (create_coze_room_request("pat_Zb1ES8Aps7mdcVD0zzPdTmMZndE6qEK8TlUz8VtbR9nE2SAyMlCAqd0vjxuxbxDw", room_info.bot_id, thread_room_info)) { 
                    send_log_to_main("Successfully created Coze room. Joining...");
                    
                    MyRTCEventHandler event_handler(alsa_manager.get(), thread_room_info.bot_id); // 传入bot_id
                    bytertc::IRTCVideo* rtc_engine = bytertc::createRTCVideo(thread_room_info.app_id.c_str(), &event_handler, nullptr);
                    if(!rtc_engine) { send_log_to_main("Failed to create RTC engine in thread."); g_app_state = AppState::COOLDOWN; return; }

                    rtc_engine->setAudioSourceType(bytertc::kAudioSourceTypeExternal);
                    rtc_engine->registerAudioFrameObserver(&event_handler);
                    rtc_engine->enableAudioFrameCallback(bytertc::AudioFrameCallbackMethod::kRemoteUser, {});

                    bytertc::IRTCRoom* rtc_room = rtc_engine->createRTCRoom(thread_room_info.room_id.c_str());
                    if(!rtc_room) { bytertc::destroyRTCVideo(); send_log_to_main("Failed to create RTC room in thread."); g_app_state = AppState::COOLDOWN; return; }
                    
                    rtc_room->setRTCRoomEventHandler(&event_handler);
                    event_handler.rtc_room_ = rtc_room; // 传递给事件处理器

                    bytertc::UserInfo user_info;
                    user_info.uid = thread_room_info.user_id.c_str();
                    user_info.extra_info = "";

                    bytertc::RTCRoomConfig room_config;
                    room_config.room_profile_type = bytertc::kRoomProfileTypeChat;
                    room_config.is_auto_publish = true;
                    // room_config.is_auto_subscribe 默认是 true，无需设置
                    
                    if (rtc_room->joinRoom(thread_room_info.token.c_str(), user_info, room_config) == 0) {
                        g_app_state = AppState::IN_CONVERSATION;
                        send_log_to_main("RTC thread: Joined room. Starting audio push loop.");
                        
                        const int samples_per_20ms = 960;
                        std::vector<int16_t> capture_buffer(samples_per_20ms);
                        
                        auto last_sound_time = std::chrono::steady_clock::now();
                        
                        while(g_app_state == AppState::IN_CONVERSATION && !g_should_exit) {
                            int samples = alsa_manager->read(capture_buffer.data(), samples_per_20ms);
                            if(samples > 0) {
                                double current_volume = calculate_rms(capture_buffer.data(), samples);
                                // 线程内发送日志，主线程会去拉取
                                send_log_to_main("Volume: " + std::to_string(static_cast<int>(current_volume)));
                                
                                if (current_volume > 500) { // 持续对话的音量检测，避免纯静音上传
                                    last_sound_time = std::chrono::steady_clock::now();
                                }
                                
                                bytertc::AudioFrameBuilder builder;
                                builder.sample_rate = bytertc::kAudioSampleRate48000;
                                builder.channel = bytertc::kAudioChannelMono;
                                builder.data = reinterpret_cast<uint8_t*>(capture_buffer.data());
                                builder.data_size = samples * sizeof(int16_t);
                                
                                bytertc::IAudioFrame* audio_frame = bytertc::buildAudioFrame(builder);
                                if (audio_frame) {
                                    rtc_engine->pushExternalAudioFrame(audio_frame);
                                    audio_frame->release();
                                }
                            }
                            
                            if (std::chrono::steady_clock::now() - last_sound_time > std::chrono::seconds(10)) { // 延长静音时间
                                send_log_to_main("No sound detected for 10 seconds, conversation ending.");
                                break; 
                            }
                            
                            std::this_thread::sleep_for(std::chrono::milliseconds(20));
                        }
                    } else {
                         send_log_to_main("RTC thread: Failed to join room!"); // 简化错误信息，具体错误由create_coze_room_request打印
                    }
                    
                    if(rtc_room) { rtc_room->leaveRoom(); rtc_room->destroy(); }
                    if(rtc_engine) { rtc_engine->registerAudioFrameObserver(nullptr); bytertc::destroyRTCVideo(); }

                    g_app_state = AppState::COOLDOWN; 
                } else { // create_coze_room_request 失败的情况
                    send_log_to_main("Failed to create Coze room from thread. Moving to COOLDOWN state."); // 修正日志信息
                    g_app_state = AppState::COOLDOWN;
                }
            });
            create_room_and_rtc_thread.detach(); // 分离线程

        } else if (g_app_state == AppState::COOLDOWN) {
            std::cout << "\nCooldown for 5 seconds before listening again..." << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(5));
            g_app_state = AppState::WAITING_FOR_TRIGGER;
            std::cout << "\nListening for loud sound to trigger..." << std::endl;
        } else {
             std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    alsa_manager->release();
    std::cout << "Application exited cleanly." << std::endl;
    return 0;
}
