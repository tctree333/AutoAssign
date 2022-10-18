import pulp

with open("data.csv", "r") as f:
    data = [row.strip().split(",") for row in f.readlines()]

events = data[0][1:]
people = [x[0] for x in data[1:]]

print("Generating pairs")
pairs = [
    (row[0], events[i], float(score))
    for row in data[1:]
    for i, score in enumerate(row[1:])
    if score != ""
]
print(pairs[:5])

x = pulp.LpVariable.dicts("pairs", pairs, 0, 1, pulp.LpInteger)

assignment_model = pulp.LpProblem("Assignment", pulp.LpMaximize)

print("Setting constraints")

# cost function
assignment_model += pulp.lpSum([x[p] * p[2] for p in pairs])

# There can only be 15 people per team
assignment_model += pulp.lpSum([x[p] for p in pairs]) == 46

# Each person must have at least 3 events
for person in people:
    assignment_model += (
        pulp.lpSum([x[p] for p in pairs if person == p[0]]) >= 3,
        "Event_minimum_%s" % person,
    )

# Each event has 2 people
for event in events:
    assignment_model += (
        pulp.lpSum([x[p] for p in pairs if event == p[1]]) == 2,
        "%s_person_count" % event,
    )

print("Solving...")

assignment_model.solve()

print("Done!")
for p in pairs:
    if x[p].value() == 1:
        print(p[0], p[1])
