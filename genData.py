import random

events = [
    "Anatomy and Physiology",
    "Astronomy",
    "Bridge",
    "Cell Biology",
    "Chem Lab",
    "Codebusters",
    "Detector Building",
    "Disease Detectives",
    "Dynamic Planet",
    "Environmental Chemistry",
    "Experimental Design",
    "Fermi Questions",
    "Flight",
    "Forensics",
    "Forestry",
    "Green Generation",
    "It's About Time",
    "Remote Sensing",
    "Rocks and Minerals",
    "Scrambler",
    "Trajectory",
    "Wifi Lab",
    "Write It Do It",
]

people = [
    "Adrian",
    "Alex",
    "Andy",
    "Ben",
    "Beth",
    "Bobby",
    "Caleb",
    "Cameron",
    "Carl",
    "Dana",
    "David",
    "Dylan",
    "Ethan",
    "Evan",
    "Frank",
    "Gabe",
    "Garrett",
    "Gavin",
    "Hannah",
    "Harrison",
    "Ian",
    "Isaac",
    "Jack",
    "Jacob",
    "James",
    "Jason",
    "Jasper",
    "Kevin",
    "Kris",
    "Logan",
    "Luke",
    "Matt",
    "Michael",
    "Nathan",
    "Nick",
    "Owen",
    "Peter",
    "Quinn",
    "Riley",
    "Sam",
    "Seth",
    "Travis",
    "Tyler",
    "Uriah",
    "Vincent",
    "Wesley",
    "William",
    "Xavier",
    "Zach",
]

header = ["Name"] + events
output = [header]
for name in people:
    row = [name] + ([""] * len(events))
    for _ in range(8):
        row[random.randint(1, len(events))] = f"{random.random()*100:.2f}"

    output.append(row)

with open("data.csv", "w") as f:
    for row in output:
        f.write(",".join(row) + "\n")
