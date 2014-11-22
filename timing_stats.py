import time
import numpy as np


class TimingStats(object):
    def __init__(self, timer_list=[], multiplier=1000):
        assert(multiplier > 0)
        self.multiplier = multiplier

        # Tracks start times for each timer
        self.timers = {'default': 0}

        # Keeps track of the difference between the start time and end time
        # (i.e. end - start) each time the timer is started and stopped.
        # Analogous to laps on stopwatches.
        self.diffs = {'default': []}

        # Sum of the differences, i.e. the total time the timer ran, between
        # timer resets.
        self.accums = {'default': 0}

        # A count of the number of times the timer was started and stopped
        # between resets.
        self.accum_counts = {'default': 0}

        # Keeps a history of the accumulated values, i.e. the total amount of
        # time each timer was ran and the number of times it was accumulated
        # between resets. The list of differnces used in conjuction with the
        # accumulation counts from this could be used to reconstruct the
        # total run times between resets in this list. As such this simply
        # acts as a cache for those values.
        self.accumed = {'default': []}

        # Create specified timers
        self.add(timer_list)

    def add(self, timer_list):
        assert(isinstance(timer_list, list))
        for timer in timer_list:
            self.timers[timer] = 0
            self.diffs[timer] = []
            self.accums[timer] = 0
            self.accum_counts[timer] = 0
            self.accumed[timer] = []

    def start(self, timer='default'):
        # Record the start time
        start_time = time.time()
        self.timers[timer] = start_time

    def end(self, timer='default'):
        # Record the difference between the start and end time
        end_time = time.time()
        diff = end_time*self.multiplier - self.timers[timer]*self.multiplier
        if timer not in self.diffs.keys():
            self.diffs[timer] = []
        self.diffs[timer].append(diff)

        # Add the difference to the accumulator and increment the related count
        self.accums[timer] = (self.accums[timer] + diff)
        self.accum_counts[timer] = self.accum_counts[timer] + 1

        return diff

    def reset(self, accum=None):
        # Store the accumulated time and count for the specified timer or all
        # timers if none was specified and then reset the associated
        # accumulator value and count.
        if accum is not None:
            if accum not in self.accumed.keys():
                self.accumed[accum] = []
            self.accumed[accum].append(
                (self.accum_counts[accum], self.accums[accum])
            )
            self.accums[accum] = 0
            self.accum_counts[accum] = 0
        else:
            for accum, value in self.accums.iteritems():
                if accum not in self.accumed.keys():
                    self.accumed[accum] = []
                self.accumed[accum].append(
                    (self.accum_counts[accum], self.accums[accum])
                )
                self.accums[accum] = 0
                self.accum_counts[accum] = 0

    def mean_difference(self, timer='default'):
        return np.mean(self.diffs[timer])

    def __str__(self):
        summary = ''
        for accum, total_time in self.accums.iteritems():
            timing = total_time
            denom = self.accum_counts[accum]
            if denom > 0:
                timing = timing/denom
            summary = "%s %s: %f, " % (summary, accum, timing)
        return summary[1:-2]

if __name__ == '__main__':
    from time import sleep

    ts = TimingStats(timer_list=['A', 'B'])
    ts.start('A')
    sleep(0.1)
    ts.end('A')
    print ts.accumed
    print ts.diffs
    print ts

    ts.reset('A')
    print ts.accumed
    print ts.diffs
    print ts

    ts.start('A')
    sleep(0.1)
    ts.end('A')

    ts.start('A')
    sleep(0.1)
    ts.end('A')

    ts.start('A')
    sleep(0.1)
    ts.end('A')
    print ts.accumed
    print ts.diffs
    print ts

    ts.reset('A')
    print ts.accumed
    print ts.diffs
    print ts
