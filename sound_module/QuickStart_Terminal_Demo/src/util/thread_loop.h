
#include <thread>
#include <atomic>
#include <iostream>

#include <functional>
#include <iostream>
#include <algorithm>

#include <queue>
#include <vector>
#include <unordered_set>
#include <mutex>
#include <condition_variable>
#ifdef __Linux__
#include <sys/epoll.h>
#include <sys/time.h>
#include <unistd.h>
#endif
#include "util.h"

#pragma once


class Timer {
  public:
    Timer(int interval, std::function<void(void)> fun, bool is_oneshot = false)
        : interval_(interval), fun(fun), is_oneshot_(is_oneshot) {
        now_ = bytertc::getCurrentMillisecs();
        count_ = 0;
        updateExpire();
    }

    inline void active() { 
        fun();
    }

    inline unsigned long long getExpire() const{ 
        return expire_; 
    }

    inline void updateExpire() {
        ++ count_;
        expire_ = now_ + count_ * interval_;
    };

    inline bool isOneshot() {
        return is_oneshot_; 
    }

  private:
    std::function<void(void)> fun;
    bool is_oneshot_;
    int count_ = 0;
    int interval_ = 0;
    unsigned long long now_; 
    unsigned long long expire_;
};


class TimerManager
{
public:
    TimerManager() {
    }
    ~TimerManager() {
        while(!queue_.empty())  {
            Timer* top = queue_.top();
            queue_.pop();
            delete top;
            top = nullptr;
        }
    }

    Timer *addTimer(int timeout, std::function<void(void)> fun, bool is_oneshot) {
        if(timeout <= 0)
            return NULL;
        Timer* timer = new Timer(timeout, fun, is_oneshot);

        queue_.push(timer);
        return timer;
    }

    void delTimer(Timer* timer)
    {
        std::priority_queue<Timer*,std::vector<Timer*>,cmp> newqueue;

        while( !queue_.empty() )
        {
            Timer* top = queue_.top();
            queue_.pop();
            if( top != timer )
                newqueue.push(top);
        }

        queue_ = newqueue;
    }

    unsigned long long getRecentTimeout()
    {
        long long timeout = -1;
        if( queue_.empty() ) {
            std::cout << " [CLI] " << " getRecentTimeout: " << " queue_.empty() " << std::endl;
            return timeout;
        }

        unsigned long long now = bytertc::getCurrentMillisecs();
        long long expire_time = queue_.top()->getExpire();
        if (expire_time <= now) {
            return 0;
        }
        timeout = expire_time - now;
        return timeout;
    }

    void takeAllTimeout() {   
        unsigned long long now = bytertc::getCurrentMillisecs();
        while ( !queue_.empty() ) {
            Timer* timer = queue_.top();
            if( timer->getExpire() <= now ) {
                queue_.pop();
                timer->active();

                if (!timer->isOneshot()) {
                    timer->updateExpire();
                    queue_.push(timer);
                } else {
                    delete timer;
                }
                continue;
            }
            return;
        }
    }

private:

    struct cmp
    {
        bool operator()(Timer*& lhs, Timer*& rhs) const { return lhs->getExpire() > rhs->getExpire(); }
    };

    std::priority_queue<Timer*,std::vector<Timer*>,cmp> queue_;
};


class ThreadLoop {
public:
    // The Task
    typedef std::function< void(void) >             task_job_t;
    ThreadLoop() {
    };
    ~ThreadLoop() {
    };

    ThreadLoop(const ThreadLoop& ) = delete;
    ThreadLoop& operator=(const ThreadLoop& ) = delete;

    void do_loop() {
        std::cout << " [CLI] " << " do loop begin" << std::endl;
        std::thread loop_([&](){
            std::lock_guard<std::mutex> _(mutex_);
            while(run_) {
                auto timeout = time_manager_.getRecentTimeout();
                //TODO: use epoll to push video and audio in one thread

                if (timeout > 0) {
                    std::unique_lock<std::mutex> lock(run_lock_);
                    cv_.wait_for(lock, std::chrono::microseconds(timeout * 1000), []() { return false;} );
                }
                time_manager_.takeAllTimeout();
            }
        });
        loop_.detach();
    }

    void cancel_loop() {
        run_ = false;
        std::lock_guard<std::mutex> _(mutex_);
    }

    Timer *addTimer(int timeout, std::function<void(void)> fun, bool is_oneshot) {
        return time_manager_.addTimer(timeout, fun, is_oneshot);
    }

    void delTimer(Timer* timer) {
        return time_manager_.delTimer(timer);
    }

private:
    std::mutex mutex_;
    std::atomic<bool> run_ {true};

    TimerManager time_manager_;

    std::condition_variable cv_{};
    std::mutex run_lock_;
};
