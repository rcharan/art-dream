import time
from watchdog.observers import Observer


class Watcher:
    def __init__(self, directory_to_watch, handler, name):
        self.directory_to_watch = directory_to_watch
        self.observer = Observer()
        self.event_handler = handler
        self.name = name

    def run(self):
        self.observer.schedule(self.event_handler, self.directory_to_watch, recursive=True)
        self.observer.start()
        try:
            print(f'\n-----------------------------------------\n'
                  f'Watcher {self.name} all set up')
            while True:
                time.sleep(5)
        except KeyboardInterrupt:
            self.observer.stop()
            print(f"\n Watcher {self.name} terminated by user")
        except:
            self.observer.stop()
            print(f'Unknown error in watcher {self.name}')

        self.observer.join()
