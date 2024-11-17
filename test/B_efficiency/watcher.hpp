#ifndef WATCHER_HPP
#define WATCHER_HPP


#include "vector.hpp"
#include <Arduino.h>

class Watcher
{
  private:
    uint64_t dt_watcher=0, _start=0;
    char *name;
    static Vector<Watcher *> watchers;
  public:
    Watcher(char *name);
    void start(){_start = esp_timer_get_time();}
    void stop(){dt_watcher += esp_timer_get_time() - _start;}
    void reset(){dt_watcher = 0;}
    uint64_t get() const{return dt_watcher;}
    template <typename T, typename... Args> T watch(T(Func)(Args...), Args... args);
    template <typename... Args>
    void watchv(void(Func)(Args...), Args... args);
    void print();
    static void printAll();
    static void resetAll();
};



#endif