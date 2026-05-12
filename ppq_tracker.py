# import modules
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from datetime import timedelta
import matplotlib.dates as mdates
import math
import pandas as pd

# Inputs class
class Inputs:
    """Stores defaults values for the script."""
    # log file location
    log_file = "ppq_log.json"

    # exam information
    exam_year = "2026"

    # plotting parameters
    figsize = (12, 5)
    fontsize = 12
    titlesize = 14
    ratingsize = 9
    grid_alpha = 0.5

    # tag-colours dictionary
    colours = {
        "4A3":  [1, 0.6, 0.0],
        "4A9":  [0.6, 0.1, 0.6],
        "4A13": [1.0, 0.85, 0.0],
        "4A15": [1.0, 0.6, 0.9],
        "4C8":  [0.7, 0.7, 0.7],
        "4E6":  [0.0, 0.6, 0.0]
    }

# Colours class
class Colours:
    """Class used to store ANSI escape sequences for printing colours."""
    # store ASCII codes for selected colours as class attributes
    RED = '\033[91m'
    ORANGE = '\033[38;5;208m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    CYAN = '\033[96m'
    PURPLE = '\033[95m'
    BLUE = '\033[94m'
    PINK = '\033[38;5;212m'
    GREY = '\033[90m'
    _END = '\033[0m'

# Logger class
class Logger:
    """Stores and handles log entries provided by the user."""
    def __init__(self):
        """Creates an instance of the Logger class."""
        # log file already exists
        if os.path.exists(Inputs.log_file):

            # open log file
            with open(Inputs.log_file, "r") as f:

                # load data
                self.data = json.load(f)
                print(f"Loaded log from {Colours.GREEN}{Inputs.log_file}{Colours._END}!")

        # no log file exists
        else:

            # create empty dictionary
            self.data = {}
            print(f"No existing log found — starting fresh.")

    def write(self):
        """Overwrites the current .json log file with self.data."""
        # open log file
        with open(Inputs.log_file, "w") as f:

            # write to file
            json.dump(self.data, f, indent = 2)

    def log(self, tag, year, question, rating = None):
        """Adds a new log entry to the database."""
        # tag is new
        if tag not in self.data:

            # create dictionary associated with tag
            self.data[tag] = {}
            print(f"  New tag {Colours.GREEN}{tag}{Colours._END} created.")

        # year is new
        if year not in self.data[tag]:

            # create dictionary associated with year
            self.data[tag][year] = {}
        
        # question has already been submitted
        #if question in self.data[tag][year]:

            # do nothing
            #print(f"  Entry {tag} | {year} | {question} already exists — skipping.")

            # change value of question
            #question = question + "~"

        while question in self.data[tag][year]:
            question = question + "~"
        
        # question is new
        else:

            # log question and store metadata
            self.data[tag][year][question] = {
                "date": datetime.now().date().isoformat(),
                "time": datetime.now().strftime("%H:%M:%S"),
                "rating": rating
            }

            # write to database
            self.write()
            print(
                f"  Logged: {Colours.GREEN}{tag}{Colours._END} | {Colours.GREEN}{year}"
                f"{Colours._END} | {Colours.GREEN}{question}{Colours._END}"
                + (f" | {rating}" if rating else "")
            )

    def remove(self, tag, year, question):
        """Removes a log entry from the database."""
        # try-except block
        try:

            # delete question dictionary
            del self.data[tag][year][question]

            # year dictionary is now empty
            if not self.data[tag][year]:

                # delete year dictionary
                del self.data[tag][year]

            # tag dictionary is now empty
            if not self.data[tag]:

                # delete tag dictionary
                del self.data[tag]

            # write to database
            self.write()
            print(
                f"  Removed: {Colours.RED}{tag}{Colours._END} | {Colours.RED}{year}{Colours._END}"
                f" | {Colours.RED}{question}{Colours._END}"
            )

        # catch errors
        except KeyError:

            # do nothing
            print(f"  Entry {tag} | {year} | {question} not found — nothing removed.")

    def show(self, display_tag = None):
        """Displays the contents of the log file in the terminal."""
        # log is empty
        if not self.data:

            # print user feedback
            print("  Log is empty.")
            return
        
        # create empty colours dictionary
        colours = {}

        # get list of ASCII codes which repeats if there are more tags than unique colours
        ascii_codes = [getattr(Colours, i) for i in Colours.__dict__.keys() if i[:1] != '_']
        ascii_codes = (ascii_codes * math.ceil(len(self.data) / len(ascii_codes)))[:len(self.data)]
        
        # associate each tag with a colour
        for tag, ascii_code in zip(sorted(self.data), ascii_codes):

            # assign colour to tag
            colours[tag] = ascii_code

        # calculate lengths of longest tag, year, and question strings
        tag_length = max(len(x) for x in sorted(self.data))
        year_length = max(len(x) for y in self.data.values() for x in y)
        question_length = max(len(x) for z in self.data.values() for y in z.values() for x in y)
        
        # get list of tags and loop
        tags = [display_tag] if display_tag else sorted(self.data)
        for tag in tags:

            # tag does not exist
            if tag not in self.data:

                # print user feedback
                print(f"  Tag {Colours.RED}{tag}{Colours._END} not found.")
                continue

            # store shorthands for colour codes
            c = colours[tag]
            e = Colours._END

            # print tag and loop for each year stored under that key
            for i, year in enumerate(sorted(self.data[tag])):

                # loop for each question stored for that year
                for j, (question, meta) in enumerate(sorted(self.data[tag][year].items())):

                    # create tag column of string
                    string = f"  [{c}{tag}{e}]" if i == 0 and j == 0 else " " * (len(tag) + 4)
                    string += " " * (tag_length - len(tag))

                    # append year column of string
                    string += f"  [{year}]" if j == 0 else " " * (len(year) + 4)
                    string += " " * (year_length - len(year))

                    # append question number
                    string += f"  {question}" if "~" not in question else " " * (len(question) + 2)
                    string += " " * (question_length - len(question))

                    # append metadata
                    string += f"  ({meta['date']})"

                    # append rating if it has been given
                    string += f"  {meta['rating']}" if meta.get("rating") else ""

                    # print constructed string
                    print(string)

    def plot(self):
        """Creates a summary plot of the database."""
        # create plot
        fig, ax = plt.subplots(figsize = Inputs.figsize)

        # get list of all dates spanning the range
        dates = pd.date_range(
            pd.Timestamp(min(
                self.data[tag][year][question]["date"] for tag in self.data
                for year in self.data[tag] for question in self.data[tag][year]
            )) - timedelta(days = 1),
            pd.Timestamp(max(
                self.data[tag][year][question]["date"] for tag in self.data
                for year in self.data[tag] for question in self.data[tag][year]
            )),
            freq="D"
        ).to_list()

        # convert to list of date indices
        date_strings = [d.strftime("%Y-%m-%d") for d in dates]
        date_index = {date: index for index, date in enumerate(date_strings)}

        # collect and sort all entries by date
        entries = [
            {
                "tag": tag,
                "year": year,
                "question": question,
                "date": self.data[tag][year][question]["date"],
                "time": self.data[tag][year][question]["time"],
                "rating": self.data[tag][year][question]["rating"],
            }
            for tag in self.data
            for year in self.data[tag]
            for question in self.data[tag][year]
        ]
        entries = sorted(entries, key = lambda entry: (entry["date"], entry["time"]))

        # initialise counts array to track bottom of bar chart
        counts = np.zeros(len(dates))

        # loop for each entry
        for entry in entries:

            # get index corresponding to date
            index = date_index[entry["date"]]

            # add segment to bar chart
            ax.bar(
                dates[index], 1, bottom=counts[index], color=Inputs.colours[entry["tag"]],
                hatch="///" if entry["year"] == Inputs.exam_year else "",
                edgecolor = "grey" if entry["year"] == Inputs.exam_year else "none",
                label=entry["tag"]
            )

            # increment per-date entries counter
            counts[index] += 1

            # add rating text
            rating = (
                entry["rating"] if len(entry["rating"]) < 4
                else entry["rating"][:3] + "\n" + entry["rating"][3:]
            )
            ax.text(
                dates[index], counts[index] - 0.62, rating,
                ha = "center", va = "center", fontsize = Inputs.ratingsize,
                linespacing = 0.5
            )

        # loop for each tag
        """for tag in self.data:

            # copy counts array to set bottom of stacked bar chart
            bottom = counts.copy()

            # loop for each year
            for year in self.data[tag]:

                # loop for each question
                for question in self.data[tag][year]:

                    # count date entry
                    index = date_index[self.data[tag][year][question]["date"]]
                    counts[index] += 1

                    # add rating text
                    ax.text(
                        dates[index], counts[index] - 0.62, self.data[tag][year][question]["rating"],
                        ha = "center", va = "center", fontsize = Inputs.titlesize
                    )

            # add tag to bar chart
            ax.bar(
                dates, counts - bottom, bottom = bottom, color = Inputs.colours[tag], label = tag
            )"""

        # get cumulative totals
        totals = list(np.cumsum(counts))

        # create line chart
        twin = ax.twinx()
        twin.plot(dates, totals, color = "C0", linewidth = 5, alpha = 0.7)

        # set y-axis limits
        ax.set_ylim(0, max(counts) * 1.05)
        """factor = max(totals) / max(counts)
        factor /= 5
        factor = math.ceil(factor)
        factor *= 5
        twin.set_ylim(0, factor * max(counts) * 1.05)"""
        ticks = ax.get_yticks()
        first_tick = next(t for t in ticks if t > 0)
        factor = max(totals) / max(counts) * first_tick
        magnitude = 10 ** math.floor(math.log10(factor))
        normalised = factor / magnitude
        if normalised <= 1: nice = 1
        elif normalised <= 2: nice = 2
        elif normalised <= 2.5: nice = 2.5
        elif normalised <= 5: nice = 5
        else: nice = 10
        factor = nice * magnitude
        twin.set_ylim(0, factor * max(counts) / first_tick * 1.05)

        # set y-axis labels
        ax.set_ylabel("Daily Total", fontsize = Inputs.titlesize)
        twin.set_ylabel("Running Total", fontsize = Inputs.titlesize)

        # colour y-axis labels
        ax.yaxis.label.set_color("k")
        twin.yaxis.label.set_color("C0")

        # colour y-axis ticks
        ax.tick_params(axis = "y", colors = "k")
        twin.tick_params(axis = "y", colors = "C0")

        # format the x-axis to show dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())

        # configure x-ticks
        fig.autofmt_xdate()
        ax.set_xticks(dates)
        ax.set_xticklabels([d.strftime("%Y-%m-%d") for d in dates])

        # set x- and y-tick font size
        ax.tick_params(axis = "both", which = "major", labelsize = Inputs.fontsize)
        twin.tick_params(axis = "both", which = "major", labelsize = Inputs.fontsize)

        # add bar chart legend
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(
            unique.values(), unique.keys(),
            fontsize=Inputs.fontsize,
            loc='center', bbox_to_anchor=(0.93, 0.5), bbox_transform=fig.transFigure
        )

        # add grid lines
        ax.grid(axis = "y", alpha = Inputs.grid_alpha)

        # tight layout and show plot
        fig.tight_layout(rect=[0, 0, 0.9, 1])
        plt.show()

    def pie(self):
        """Creates a pie chart of the database."""
        # create plot
        fig, ax = plt.subplots(figsize=Inputs.figsize)

        # count entries per tag
        tag_counts = {}
        unique_counts = {}
        for tag in self.data:
            count = sum(
                1
                for year in self.data[tag]
                for question in self.data[tag][year]
            )
            if count > 0:
                tag_counts[tag] = count
            unique_count = sum(
                1
                for year in self.data[tag]
                for question in self.data[tag][year]
                if "~" not in question
            )
            if count > 0:
                unique_counts[tag] = unique_count

        tags = list(tag_counts.keys())
        counts = list(tag_counts.values())
        unique_counts = list(unique_counts.values())
        colours = [Inputs.colours[tag] for tag in tags]

        # draw pie chart
        wedges, _ = ax.pie(
            counts,
            labels=None,
            colors=colours,
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5}
        )

        # overlay entry counts on each slice
        #total = sum(counts)
        for wedge, count, unique_count in zip(wedges, counts, unique_counts):
            angle = (wedge.theta1 + wedge.theta2) / 2
            angle_rad = np.deg2rad(angle)
            radius = 0.6
            x = radius * np.cos(angle_rad)
            y = radius * np.sin(angle_rad)
            ax.text(
                x, y, f"{count}" if count == unique_count else f"{count} ({unique_count})",
                ha="center", va="center",
                fontsize=Inputs.titlesize,
                fontweight="bold",
                color="white"
            )

        # add legend
        ax.legend(
            wedges, tags,
            fontsize = Inputs.fontsize,
            loc = "best"
        )

        fig.tight_layout()
        plt.show()

def parse_entry(tokens):
    """
    Parses [tag] [year] Q[number] (rating)? from a list of string tokens.
    Returns (tag, year, question, rating) or raises ValueError.
    """
    if len(tokens) not in (3, 4):
        raise ValueError("Expected: [command] [tag] [year] Q[number] (rating)?")
    tag, year, question = tokens[:3]
    rating = tokens[3] if len(tokens) == 4 else None
    if not question.upper().startswith("Q"):
        raise ValueError(f"Question must start with 'Q' (got '{question}')")
    if rating is not None and not all(c == "*" for c in rating):
        raise ValueError(f"Rating must be stars only e.g. '***' (got '{rating}')")
    return tag, year, question.upper(), rating

def print_help():
    """Prints help instructions to the user on request."""
    # print each command with explanation
    print("  Commands:")
    print("    log  [tag] [year] Q[n]  — add an entry")
    print("    rm   [tag] [year] Q[n]  — remove an entry")
    print("    show [tag]              — display logged entries")
    print("    plot [tag]              — display summary plot")
    print("    pie  [tag]              — display pie chart")
    print("    help                    — show this message")
    print("    quit / exit             — exit the script")

# main function
def main():

    # create instance of Logger class
    logger = Logger()
    print(f"Past paper tracker ready. Type {Colours.CYAN}help{Colours._END} for usage.\n")

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not raw:
            continue

        tokens = raw.split()
        command = tokens[0].lower()

        if command in ("quit", "exit"):
            print("Exiting.")
            break

        elif command == "help":
            print_help()

        elif command == "log":
            try:
                tag, year, question, rating = parse_entry(tokens[1:])
                logger.log(tag, year, question, rating)
            except ValueError as e:
                print(f"  Error: {e}")

        elif command == "rm":
            try:
                tag, year, question, rating = parse_entry(tokens[1:])
                logger.remove(tag, year, question)
            except ValueError as e:
                print(f"  Error: {e}")

        elif command == "show":
            tag_filter = tokens[1] if len(tokens) > 1 else None
            logger.show(tag_filter)

        elif command == "plot":

            logger.plot()

        elif command == "pie":

            logger.pie()

        else:
            print(f"  Unknown command '{command}'. Type 'help' for usage.")

# upon script execution
if __name__ == "__main__":

    # run main
    main()

    # show all plots
    #plt.show()