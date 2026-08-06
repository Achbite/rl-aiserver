#include "session/session_manager.h"
#include "log/logger.h"

// ---- 创建新会话 ----
std::string SessionManager::CreateSession() {
    std::lock_guard<std::mutex> lock(mutex_);

    const std::string sid = "maze-session-" +
                            std::to_string(next_session_id_++);
    sessions_[sid] = Session{};
    sessions_[sid].session_id = sid;

    LOG_INFO("SessionManager", "创建会话 session_id=%s, 活跃会话数=%zu",
             sid.c_str(), sessions_.size());

    return sid;
}

// ---- 获取指定会话 ----
SessionManager::Session* SessionManager::GetSession(
    const std::string& session_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) {
        return nullptr;
    }
    return &it->second;
}

// ---- 销毁指定会话 ----
void SessionManager::DestroySession(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = sessions_.find(session_id);
    if (it != sessions_.end()) {
        sessions_.erase(it);
        LOG_INFO("SessionManager", "销毁会话 session_id=%s, 剩余会话数=%zu",
                 session_id.c_str(), sessions_.size());
    }
}

// ---- 获取当前活跃会话数 ----
int SessionManager::GetActiveSessionCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    int count = 0;
    for (const auto& item : sessions_) {
        if (item.second.session_state != maze::SESSION_STATE_CLOSED) {
            ++count;
        }
    }
    return count;
}

int SessionManager::GetActiveEpisodeCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    int count = 0;
    for (const auto& item : sessions_) {
        if (item.second.episode_state == EpisodeState::Active) {
            ++count;
        }
    }
    return count;
}

std::vector<std::string> SessionManager::GetSessionIds() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> result;
    result.reserve(sessions_.size());
    for (const auto& item : sessions_) result.push_back(item.first);
    return result;
}
