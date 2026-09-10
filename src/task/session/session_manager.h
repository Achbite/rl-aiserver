#pragma once

#include "log/logger.h"
#include "proto/communication/session.pb.h"

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// Task state remains strongly typed while session storage owns its lifecycle.
template <class SessionType>
class SessionManager {
public:
    using Session = SessionType;
    using AgentRuntime = typename Session::AgentRuntime;

    struct ClientActivitySnapshot {
        int active_session_count = 0;
        int recent_active_session_count = 0;
        int64_t latest_active_activity_unix_ms = 0;
    };

    std::string CreateSession() {
        std::lock_guard<std::mutex> lock(mutex_);
        const std::string sid = "session-" + std::to_string(next_session_id_++);
        sessions_[sid] = Session{};
        sessions_[sid].session_id = sid;
        LOG_INFO("SessionManager", "创建会话 session_id=%s, 活跃会话数=%zu",
                 sid.c_str(), sessions_.size());
        return sid;
    }

    Session* GetSession(const std::string& session_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = sessions_.find(session_id);
        return it == sessions_.end() ? nullptr : &it->second;
    }

    void DestroySession(const std::string& session_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = sessions_.find(session_id);
        if (it != sessions_.end()) {
            sessions_.erase(it);
            LOG_INFO("SessionManager", "销毁会话 session_id=%s, 剩余会话数=%zu",
                     session_id.c_str(), sessions_.size());
        }
    }

    int GetActiveSessionCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        int count = 0;
        for (const auto& item : sessions_) {
            if (item.second.phase != rl::session::v1::SESSION_PHASE_CLOSED) {
                ++count;
            }
        }
        return count;
    }

    ClientActivitySnapshot GetClientActivitySnapshot(
        int64_t now_unix_ms, int64_t lease_ms) const {
        std::lock_guard<std::mutex> lock(mutex_);
        ClientActivitySnapshot snapshot;
        for (const auto& item : sessions_) {
            const auto& session = item.second;
            if (session.phase == rl::session::v1::SESSION_PHASE_CLOSED) continue;
            ++snapshot.active_session_count;
            snapshot.latest_active_activity_unix_ms = std::max(
                snapshot.latest_active_activity_unix_ms,
                session.last_valid_client_activity_unix_ms);
            if (lease_ms >= 0 && session.last_valid_client_activity_unix_ms > 0 &&
                now_unix_ms >= session.last_valid_client_activity_unix_ms &&
                now_unix_ms - session.last_valid_client_activity_unix_ms <= lease_ms) {
                ++snapshot.recent_active_session_count;
            }
        }
        return snapshot;
    }

    int GetActiveEpisodeCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        int count = 0;
        for (const auto& item : sessions_) {
            if (item.second.phase == rl::session::v1::SESSION_PHASE_EPISODE_RUNNING ||
                item.second.phase == rl::session::v1::SESSION_PHASE_EPISODE_TERMINAL) {
                ++count;
            }
        }
        return count;
    }

    std::vector<std::string> GetSessionIds() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::string> result;
        result.reserve(sessions_.size());
        for (const auto& item : sessions_) result.push_back(item.first);
        return result;
    }

private:
    std::unordered_map<std::string, Session> sessions_;
    mutable std::mutex mutex_;
    uint64_t next_session_id_ = 1;
};
