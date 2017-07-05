import settings


class Algorithm(object):

    def __init__(self, file_name, display_name=None,
                 is_baseline=False, is_meta=False,
                 color=(0, 0, 0), line_style="-", marker=None):

        self.file_name = file_name
        if display_name is None:
            self.display_name = file_name.upper()
        else:
            self.display_name = display_name

        # algorithm category
        self.is_baseline_algorithm = is_baseline
        self.is_meta_algorithm = is_meta

        # plotting properties
        self.color = color
        self.line_style = line_style
        self.marker = marker

    def __lt__(self, other):
        return self.get_display_name() < other.get_display_name()

    def __str__(self):
        return self.get_name()

    def __repr__(self):
        return self.get_name()

    def get_name(self):
        return self.file_name

    def get_display_name(self):
        return self.display_name

    def is_baseline(self):
        return self.is_baseline_algorithm

    def is_meta(self):
        return self.is_meta_algorithm

    def get_color(self):
        return self.color

    def get_line_style(self):
        if self.line_style is not None:
            return self.line_style
        elif self.is_meta():
            return (0, (1, 1))
        else:
            return "-"

    @staticmethod
    def set_colors(algorithms, offset=0):
        for idx_a, algorithm in enumerate(algorithms):
            algorithm.color = settings.get_color(idx_a+offset)
        return algorithms

    @staticmethod
    def initialize_algorithms(meta_data, file_names_algorithms, is_baseline=False, is_meta=False):
        algorithms = []

        for file_name_algorithm in file_names_algorithms:
            acronym = meta_data[file_name_algorithm]["acronym"]
            algorithm = Algorithm(file_name=file_name_algorithm,
                                  display_name=acronym,
                                  is_meta=is_meta,
                                  is_baseline=is_baseline)
            algorithms.append(algorithm)

        return algorithms