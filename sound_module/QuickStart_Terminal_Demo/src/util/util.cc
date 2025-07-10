#include "util.h"
#include <cassert>
#include <chrono>
#include <sstream>
#include <iomanip>
#if defined(WINDOWS) || defined(WIN32) || defined(_WIN32)
#include <Windows.h>
#else
#include <pthread.h>
#include <sys/statfs.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#endif

namespace bytertc {
bool fileExisted( const std::string& path ) {
    struct stat _f_info;
    if ( stat(path.c_str(), &_f_info) != 0 ) return false;
    return !(bool)(_f_info.st_mode & S_IFDIR);
}

std::string& trim(std::string& text) {
    if (!text.empty()) {
        text.erase(0, text.find_first_not_of((" \n\r\t")));
        text.erase(text.find_last_not_of(" \n\r\t") + 1);
    }
    return text;    
}

// Check if string is start with sub string
bool StringStart(const std::string& s, const std::string& p) {
    if ( s.size() >= p.size() ) {
        for ( size_t i = 0; i < p.size(); ++i ) {
            if ( s[i] != p[i] ) return false;
        }
        return true;
    }
    return false;
}
size_t getFileSize(FILE* file) {
    fseek(file, 0, SEEK_END);
    size_t read_len = ftell(file);
    fseek(file, 0, SEEK_SET);
    return read_len;
}

size_t getFileSize(const char* filePath) {
    FILE* file = fopen(filePath, "rb");
    if (file == nullptr) {
        return 0;
    }
    size_t size = getFileSize(file);
    fclose(file);
    return size;
}

std::vector<unsigned char> readFile(const char* file_path,const char *mode) {
    FILE* file = fopen(file_path, mode);
    std::vector<uint8_t> result;
    if (file == nullptr) {
        return result;
    }

    // 获取文件大小，尽量一次读完
    size_t fileSize = getFileSize(file);
    if (fileSize != 0) {
        result.resize(fileSize);
        size_t n = fread(&result[0], 1, fileSize, file);
        assert(n <= fileSize);
        if (n != fileSize) {
            result.resize(n);
        }
    }
    // 在读取过程当中，有可能文件大小有变化，再尝试读取
    const size_t read_len = 1024;
    char buf[read_len];
    for (;;) {
        size_t n = fread(buf, 1, read_len, file);
        result.insert(result.end(), buf, buf + n);
        if (n < read_len) {
            break;
        }
    }
    fclose(file);
    return result;
}

unsigned long long getCurrentMillisecs() {
    // struct timespec ts;
    // clock_gettime(CLOCK_MONOTONIC_COARSE, &ts);
    // return ts.tv_sec * 1000 + ts.tv_nsec / (1000 * 1000);
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void printLog(const std::string & str) {
	std::cout << str;
}

unsigned long getCurrentThreadId() {

#if defined(WINDOWS) || defined(WIN32) || defined(_WIN32)
	return ::GetCurrentThreadId();
#else
	return pthread_self();
#endif
}
void printHelpAndExit()
{
	LOG_INFO("Please check if the 'config.json' file parameters are correct!");
	exit(0);
}

void setCurrentDir(const std::string & str) {
#if defined(WINDOWS) || defined(WIN32) || defined(_WIN32)
	 ::SetCurrentDirectoryA(str.c_str());
#else
	chdir(str.c_str());
#endif
}

std::string getFileName(const std::string & filePath) {
#if defined(WINDOWS) || defined(WIN32) || defined(_WIN32)
	auto index = filePath.rfind("\\");
#else
	auto index = filePath.rfind("/");
#endif
	std::string fileName = filePath;
	if (index != -1) {
		fileName = filePath.substr(index + 1, filePath.size() - index -1);
	}
	return fileName;
}

std::string getExePath()
{
#if defined(WINDOWS) || defined(WIN32) || defined(_WIN32)
	std::string strPath;
	char filepath[MAX_PATH] = { 0 };
	auto sz = ::GetModuleFileNameA(nullptr, filepath, MAX_PATH);
	if (sz == 0)
		return filepath;
	for (int n = sz - 1; n > 0; n--) {
		if (filepath[n] == char('\\')) {
			filepath[n + 1] = 0;
			break;
		}
	}
	return filepath;
#else
	char *p = NULL;
	const int len = 256;
	/// to keep the absolute path of executable's path
	char chPath[len] = { 0 };
	int n = readlink("/proc/self/exe", chPath, len);
	if (NULL != (p = strrchr(chPath, '/')))
		*p = '\0';
	else
	{
		return std::string("");
	}
	return std::string(chPath);
#endif
}

std::string getCurrentTimeStr() {
#if defined(WINDOWS) || defined(WIN32) || defined(_WIN32)
	SYSTEMTIME wtm;
	GetLocalTime(&wtm);
	std::stringstream ss;
	ss << std::setw(2) << std::setfill('0') << wtm.wHour << ":" << std::setw(2) << std::setfill('0') << wtm.wMinute
		<< ":" << std::setw(2) << std::setfill('0') << wtm.wSecond << "." << std::setw(3) << std::setfill('0')
		<< wtm.wMilliseconds;
	return ss.str();
#else
	struct timeval tv;
	gettimeofday(&tv, NULL);
	time_t t = time(NULL);
	struct tm time_info;
	localtime_r(&t, &time_info);
	std::stringstream ss;
	ss << std::setw(2) << std::setfill('0') << time_info.tm_hour << ":" << std::setw(2) << std::setfill('0')
		<< time_info.tm_min << ":" << std::setw(2) << std::setfill('0') << time_info.tm_sec << "." << std::setw(3)
		<< std::setfill('0') << (tv.tv_usec / 1000);
	return ss.str();
#endif
}

int safeStrToInt(const std::string &str) 
{
	if (str.empty()) return 0;
	if (isdigit(str.front())) {
		return std::stoi(str);
	}
	return 0;
}

}

