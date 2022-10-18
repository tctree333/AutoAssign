# SciOly Team Auto-Assigner

This is a tool to automatically create optimal team or event assignments for Science Olympiad by optimizing an objective function.

The best working version is `assign_or2.py`, which uses OR-Tools. The other files do not work as well and are only included just because.

## Usage

Create a CSV file with the students, events, and data you want to optimize. This could be generated from historical performance data, event preferences, or other factors. Make sure the correct CSV file name is in the program, and modify any constraints if necessary.

In `assign_or2.py`, you can indicate which students are able to take on a 4th event, or pin certain students to certain events if necessary.

`assign_or2.py` will run until either no solution can be found or the optimal solution is found. If no optimal solution is found, you can stop execution when the objective function is sufficiently low and results will be written to disk.

## Limitations

Due to constraints with OR-Tools, there isn't really a way to say "only assign events to 15 students" if you're trying to use this to create teams. However, this program works well for event assignments once the students on a team have been determined.
