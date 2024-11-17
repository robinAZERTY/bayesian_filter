
#include "watcher.hpp"
Vector<Watcher *> Watcher::watchers;

Watcher::Watcher(char *name) : name(name)
{
    Serial.println(name);
    watchers.push_back(this);
}

template <typename T, typename... Args>
T inline Watcher::watch(T(Func)(Args...), Args... args)
{
    const uint64_t start = esp_timer_get_time();
    T ret = Func(args...);
    dt_watcher += esp_timer_get_time() - _start;
    return ret;
}

template <typename... Args>
inline void Watcher::watchv(void (Func)(Args...), Args... args)
{
    const uint64_t start = esp_timer_get_time();
    Func(args...);
    dt_watcher += esp_timer_get_time() - _start;
}
void Watcher::print()
{
    Serial.print(name);
    Serial.print(" took ");
    Serial.print(dt_watcher);
    Serial.println(" us");

}

void Watcher::resetAll()
{
    for (int i = 0; i < watchers.size(); i++)
        watchers[i]->reset();
}
void Watcher::printAll()
{
    for (int i = 0; i < watchers.size(); i++)
        watchers[i]->print();
}