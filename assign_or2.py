from ortools.sat.python import cp_model

with open("data.csv", "r") as f:
    data = [row.strip().split(",") for row in f.readlines()]

events = data[0][1:]
people = [x[0] for x in data[1:]]
costs = [list(map(lambda y: 30 if y == "" else float(y), x[1:])) for x in data[1:]]
three_people_events = ("Codebusters", "Experimental Design")
conflicts = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (9, 10, 11),
    (12, 13, 14),
    (15, 16, 17),
    (18,),
    (19,),
    (20,),
    (21,),
    (22,),
)
fourth_ev = (
)
pinned = list(
    map(
        lambda x: (people.index(x[0]), events.index(x[1])),
        [
        ],
    )
)


def main():
    # Data
    num_workers = len(costs)
    num_tasks = len(costs[0])

    # Model
    model = cp_model.CpModel()

    # Variables
    x = []
    for i in range(num_workers):
        t = []
        for j in range(num_tasks):
            t.append(model.NewBoolVar(f"x[{i},{j}]"))
        x.append(t)

    # Constraints
    # Each worker is assigned to at most 4 tasks.
    for i in range(num_workers):
        if people[i] in fourth_ev:
            model.Add(sum(x[i][j] for j in range(num_tasks)) <= 4)
        else:
            model.Add(sum(x[i][j] for j in range(num_tasks)) <= 3)

        # model.Add(sum(x[i][j] for j in range(num_tasks)) <= 4)

        model.Add(sum(x[i][j] for j in range(num_tasks)) >= 3)

        # model.Add(sum(x[i][j] for j in range(num_tasks)) != 1)
        # model.Add(sum(x[i][j] for j in range(num_tasks)) != 2)

        for conflict in conflicts:
            model.Add(sum(x[i][j] for j in conflict) <= 1)

    # Each task is assigned to correct workers.
    for j in range(num_tasks):
        if events[j] in three_people_events:
            model.Add(sum(x[i][j] for i in range(num_workers)) == 3)
        else:
            model.Add(sum(x[i][j] for i in range(num_workers)) == 2)

    for i, j in pinned:
        model.Add(x[i][j] == 1)

    # Objective
    objective_terms = []
    for i in range(num_workers):
        for j in range(num_tasks):
            objective_terms.append(costs[i][j] * x[i][j])
    model.Minimize(sum(objective_terms))

    # Solve
    printer = SolutionPrinter()
    solver = cp_model.CpSolver()
    status = solver.Solve(model, printer)

    # Print solution.
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE")
        print(f"Total cost = {solver.ObjectiveValue()}\n")
        header = ["Name", "Count"] + events
        people_out = {p: [""] * (len(events)) for p in people}
        counts = [0] * len(events)
        for i in range(num_workers):
            for j in range(num_tasks):
                if solver.BooleanValue(x[i][j]):
                    people_out[people[i]][j] = "X"
                    counts[j] += 1
                    # print(f"{people[i]} assigned to event {events[j]}.")

        with open("output.csv", "w") as f:
            f.write(",".join(header) + "\n")
            for person, out in people_out.items():
                person_count = sum(1 for x in out if x != "")
                if person_count == 0:
                    continue
                f.write(",".join([person, str(person_count)] + out) + "\n")
            f.write(",".join(["", ""] + list(map(str, counts))) + "\n")
    else:
        print("No solution found.")


class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__solution_count = 0

    def on_solution_callback(self):
        self.__solution_count += 1
        print(f"OBJ: {self.ObjectiveValue()}")

    def solution_count(self):
        return self.__solution_count


if __name__ == "__main__":
    main()
