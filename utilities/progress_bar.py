from datetime import datetime

class ProgressBar:

    def __init__(self, total_iterations, bar_width = 50):
        self.total_iterations = total_iterations
        self.start_time       = None
        self.bar_width        = bar_width

    def start(self):
        self.update(0)

    def update(self, iterations):
        percent_progress = iterations / self.total_iterations

        # Create the bar
        progress_ticks   = int(percent_progress * self.bar_width)
        bar = '=' * progress_ticks
        bar_end = '-' * (self.bar_width - progress_ticks - 1)
        if iterations == self.total_iterations:
            bar += bar_end
        else:
            bar += '>' + bar_end[1:]
              

        if iterations == self.total_iterations:
            if self.start_time is None:
                pass
            total_time = (datetime.now() - self.start_time).seconds
            time_per_iteration = total_time / iterations
            eta = f' {time_per_iteration:.2}s per loop'
        elif self.start_time is None or iterations == 0:
            self.start_time = datetime.now()
            eta             = ''
        else:
            time_elapsed    = (datetime.now() - self.start_time).seconds
            eta             = int((1-percent_progress) * time_elapsed / percent_progress)
            eta             = f' ETA: {format_time(eta)}'

        out_str = f'\r[{bar}] {iterations}/{self.total_iterations}'
        out_str += eta

        # Be sure to clear any leftover text
        out_str += ' '*10

        if iterations == self.total_iterations:
            print(out_str)
        else:
            print(out_str, end = '')

def format_time(seconds):
    out_str = ''
    hours = seconds // 3600
    if hours > 0:
        out_str += f'{hours}:'
        seconds -= hours * 3600

    minutes = seconds // 60
    if hours != 0:
        out_str += f'{minutes:02d}:'
    elif minutes > 0:
        out_str += f'{minutes}:'
    seconds -= minutes * 60

    if minutes != 0 or hours != 0:
        out_str += f'{int(seconds):02d}'
    else:
        out_str += f'{int(seconds)}s'

    return out_str
