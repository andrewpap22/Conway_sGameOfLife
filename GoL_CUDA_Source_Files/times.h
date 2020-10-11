#ifndef _TIMES_H_
#define _TIMES_H_

#ifdef __linux__

#include <stdlib.h>
#include <sys/time.h>

typedef timeval timestamp;

inline timestamp getTime(void){
	timestamp currTime;
	gettimeofday(&currTime, NULL);
	return currTime;
}
inline float getElapsedTime(timestamp prevTime){
	timeval currTime;
	gettimeofday(&currTime, NULL);
	return (currTime.tv_sec - prevTime.tv_sec) * 1000.0f + (currTime.tv_usec - prevTime.tv_usec) / 1000.0f;
}

#else

#include <time.h>

typedef clock_t timestamp;

inline timestamp getTime(void){
	return clock();
}
inline float getElapsedTime(timestamp prevTime){
	return ((float)clock()-prevTime) / CLOCKS_PER_SEC * 1000.0f;
}
#endif

#endif
