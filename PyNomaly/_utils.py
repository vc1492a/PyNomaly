# Authors: Valentino Constantinou <vc@valentino.io>
# License: Apache 2.0

import sys
from python_utils.terminal import get_terminal_size


class Utils:
    @staticmethod
    def emit_progress_bar(progress: str, index: int, total: int) -> str:
        """
        A progress bar that is continuously updated in Python's standard
        out.
        :param progress: a string printed to stdout that is updated and later
        returned.
        :param index: the current index of the iteration within the tracked
        process.
        :param total: the total length of the tracked process.
        :return: progress string.
        """

        w, h = get_terminal_size()
        sys.stdout.write("\r")
        if total < w:
            block_size = int(w / total)
        else:
            block_size = int(total / w)
        if index % block_size == 0:
            progress += "="
        percent = index / total
        sys.stdout.write("[ %s ] %.2f%%" % (progress, percent * 100))
        sys.stdout.flush()
        return progress
