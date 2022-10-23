class Patent:
    def __init__(self, pid, title, assignee, abstract):
        self.pid = pid
        self.title = title
        self.assignee = assignee
        self.abstract = abstract

    def __str__(self):
        return "[id:'{0}', title'{1}', assignee'{2}', abstract'{3}']".format(self.pid, self.title, self.assignee,
                                                                             self.abstract)
