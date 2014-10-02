import time


class TimingStats(object):
    def __init__(self, timer_list=None, multiplier=1000):
        assert(multiplier > 0)
        self.multiplier = multiplier

        self.timers = {'default': 0}
        self.diffs = {'default': 0}
        self.accums = {'default': 0}
        self.accum_counts = {'default': 0}
        self.accumed = {'default': []}

        assert(timer_list is None
               or (isinstance(timer_list, list) and len(timer_list) > 0))

        if timer_list is not None:
            for timer in timer_list:
                self.timers[timer] = 0
                self.diffs[timer] = 0
                self.accums[timer] = 0
                self.accum_counts[timer] = 0
                self.accumed[timer] = []

    def start(self, timer='default'):
        start_time = time.time()
        self.timers[timer] = start_time

    def end(self, timer='default'):
        end_time = time.time()
        diff = end_time*self.multiplier - self.timers[timer]*self.multiplier
        self.diffs[timer] = diff
        self.accums[timer] = (self.accums[timer] + diff)
        self.accum_counts[timer] = self.accum_counts[timer] + 1

    def reset(self, accum=None):
        if accum is not None:
            self.accumed[accum].append(
                (self.accum_counts[accum], self.accums[accum])
            )
            self.accums[accum] = 0
            self.accum_counts[accum] = 0
        else:
            for accum, value in self.accums.iteritems():
                self.accumed[accum].append(
                    (self.accum_counts[accum], self.accums[accum])
                )
                self.accums[accum] = 0
                self.accum_counts[accum] = 0

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
    ts = TimingStats(timer_list=['A', 'B'])
    ts.start('A')
    ts.end('A')
    print ts.accumed
    print ts

    ts.reset('A')
    print ts.accumed
    print ts

    ts.start('A')
    ts.end('A')

    ts.start('A')
    ts.end('A')

    ts.start('A')
    ts.end('A')
    print ts.accumed
    print ts

    ts.reset('A')
    print ts.accumed
    print ts
