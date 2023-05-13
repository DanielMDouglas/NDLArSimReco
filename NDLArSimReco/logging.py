class LogManager:
    def __init__(self, network):
        self.network = network

        self.entries = []
    def log_state(self):
        """
        Create a new logEntry and write it to disk
        """
        thisEntry = LogEntry()
        self.entries.append(thisEntry)
        thisEntry.manager = self
        return 
    def rewind(self):
        """
        Recall the state of the network from a given point
        Set the weights, erase orphaned logs, resent the seed
        """
        return
    def get_loss(self):
        """
        return the loss time series
        """
        for log_entry in self.entries:
            pass
        return

class LogEntry:
    def __init__(self):
        """
        A log entry contains information about the state
        of the network during training
        Each entry should contain enough information to 
        replicate the state totally, including RNG
        """
        self.manager = None
        
    def write(self):
        """
        Write the state of the network/training proccess
        """
        return
    def erase(self):
        """
        Erase the log entry on disk and delete self
        """
        return 
