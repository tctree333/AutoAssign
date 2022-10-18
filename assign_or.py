from ortools.linear_solver import pywraplp

with open("data.csv", "r") as f:
    data = [row.strip().split(",") for row in f.readlines()]

events = data[0][1:]
people = [x[0] for x in data[1:]]
costs = [list(map(lambda y: 0 if y == "" else float(y), x[1:])) for x in data[1:]]
three_people_events = ("Codebusters", "Experimental Design")


def main():
    # Data
    num_workers = len(costs)
    num_tasks = len(costs[0])

    # Solver
    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver("SCIP")

    if not solver:
        return

    # Variables
    # x[i, j] is an array of 0-1 variables, which will be 1
    # if worker i is assigned to task j.
    x = {}
    for i in range(num_workers):
        for j in range(num_tasks):
            x[i, j] = solver.IntVar(0, 1, "")

    # Constraints
    # Each worker is assigned to the correct number of tasks.
    for i in range(num_workers):
        solver.Add(solver.Sum([x[i, j] for j in range(num_tasks)]) <= 4)
        # solver.Add(solver.Sum([x[i, j] for j in range(num_tasks)]) >= 3)

    # Each task is assigned to the correct number of workers.
    for j in range(num_tasks):
        if events[j] in three_people_events:
            solver.Add(solver.Sum([x[i, j] for i in range(num_workers)]) == 3)
        else:
            solver.Add(solver.Sum([x[i, j] for i in range(num_workers)]) == 2)

    # Objective
    objective_terms = []
    for i in range(num_workers):
        for j in range(num_tasks):
            objective_terms.append(costs[i][j] * x[i, j])
    solver.Maximize(solver.Sum(objective_terms))

    # Solve
    status = solver.Solve()

    # Print solution.
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        print("OPTIMAL" if status == pywraplp.Solver.OPTIMAL else "FEASIBLE")
        print(f"Total cost = {solver.Objective().Value()}\n")
        header = ["Name", "Count"] + events
        people_out = {p: [""] * (len(events)) for p in people}
        counts = [0] * len(events)
        for i in range(num_workers):
            for j in range(num_tasks):
                # Test if x[i,j] is 1 (with tolerance for floating point arithmetic).
                if x[i, j].solution_value() > 0.5:
                    people_out[people[i]][j] = "X"
                    counts[j] += 1
                    print(f"{people[i]} assigned to event {events[j]}.")

        with open("output.csv", "w") as f:
            f.write(",".join(header) + "\n")
            for person, out in people_out.items():
                f.write(
                    ",".join([person, str(sum(1 for x in out if x != ""))] + out) + "\n"
                )
            f.write(",".join(["", ""] + list(map(str, counts))) + "\n")
    else:
        print("No solution found.")


if __name__ == "__main__":
    main()
