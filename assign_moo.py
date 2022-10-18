import numpy as np

from pymoo.core.problem import ElementwiseProblem
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from pymoo.core.repair import Repair
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.crossover.hux import HalfUniformCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize

with open("data.csv", "r") as f:
    data = [row.strip().split(",") for row in f.readlines()]

events = data[0][1:]
people = [x[0] for x in data[1:]]
three_people_events = ("Codebusters", "Experimental Design")

print("Generating Pairs")
pairs = [
    (row[0], events[i], float(score))
    for row in data[1:]
    for i, score in enumerate(row[1:])
    if score != ""
]

print("Generating Masks")
score_mask = np.array([p[2] for p in pairs])
event_mask = {
    event: np.array([1 if p[1] == event else 0 for p in pairs]) for event in events
}
person_mask = {
    person: np.array([1 if p[0] == person else 0 for p in pairs]) for person in people
}


class MyProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(
            n_var=len(pairs),
            n_obj=1,
            n_ieq_constr=len(people) + len(events) + 1,
            # n_eq_constr=len(events) + 1,
            xl=0,
            xu=1,
            vtype=bool,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        eq_constraints = []
        ieq_constraints = []

        for event in events:
            eq_constraints.append(
                2
                * (
                    (
                        np.sum(x * event_mask[event])
                        - (3 if event in three_people_events else 2)
                    )
                    ** 2
                )
            )

        on_team = set(p[0] for i, p in enumerate(pairs) if x[i] == 1)
        eq_constraints.append(10 * ((len(on_team) - 15) ** 2))

        for person in people:
            ieq_constraints.append(
                (3 if person in on_team else 0) - np.sum(x * person_mask[person])
            )

        out["F"] = -np.sum(x * score_mask)
        out["G"] = ieq_constraints + eq_constraints
        # out["H"] = eq_constraints


class MySampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), False, dtype=bool)

        for k in range(n_samples):
            I = np.random.permutation(people)[:15]
            K = X[k, :]

            index = {event: {person: 0} for event in events for person in I}
            event_count = {event: 0 for event in events}
            for i, p in enumerate(pairs):
                if p[0] in I:
                    event_count[p[1]] += 1
                    index[p[1]][p[0]] = i

            person_assigned_count = {person: 0 for person in I}
            for event, _ in sorted(event_count.items(), key=lambda x: x[1]):
                assigned = 0
                for person, idx, score, count in sorted(
                    sorted(
                        (
                            (person, idx, pairs[idx][2], person_assigned_count[person])
                            for person, idx in index[event].items()
                        ),
                        key=lambda x: x[2],
                    ),
                    key=lambda x: max(x[3] - 2, 0),
                ):
                    if person_assigned_count[person] < 5:
                        K[idx] = True
                        person_assigned_count[person] += 1
                        assigned += 1
                    if assigned >= (3 if event in three_people_events else 2):
                        break

        return X


class BinaryCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 1)

    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, n_var = X.shape

        _X = np.full((self.n_offsprings, n_matings, problem.n_var), False)

        for k in range(n_matings):
            p1, p2 = X[0, k], X[1, k]

            both_are_true = np.logical_and(p1, p2)
            _X[0, k, both_are_true] = True

            n_remaining = 48 - np.sum(both_are_true)

            I = np.where(np.logical_xor(p1, p2))[0]

            S = I[np.random.permutation(len(I))][:n_remaining]
            _X[0, k, S] = True

        return _X


class MyMutation(Mutation):
    def _do(self, problem, X, **kwargs):
        for i in range(X.shape[0]):
            X[i, :] = X[i, :]
            for event in np.random.permutation(events):
                in_event = np.nonzero(X[i, :] * event_mask[event])[0]
                not_in_event = np.nonzero(np.logical_not(X[i, :]) * event_mask[event])[
                    0
                ]
                if len(not_in_event) > 0 and len(in_event) > 0:
                    X[i, np.random.choice(in_event)] = 0
                    X[i, np.random.choice(not_in_event)] = 1
                    break

        return X


class RepairEvents(Repair):
    def _do(self, problem, Z, **kwargs):
        for i in range(len(Z)):
            z = Z[i]

            event_count = {event: 0 for event in events}
            person_count = {person: 0 for person in people}
            for i, p in enumerate(pairs):
                if z[i] == 1:
                    event_count[p[1]] += 1
                    person_count[p[0]] += 1

            for i, x in enumerate(z):
                p = pairs[i]
                if (
                    x == 1
                    and person_count[p[0]] > 3
                    and event_count[p[1]] > (3 if p[1] in three_people_events else 2)
                ):
                    person_count[p[0]] -= 1
                    event_count[p[1]] -= 1
                    z[i] = 0
                elif (
                    x == 0
                    and person_count[p[0]] < 3
                    and event_count[p[1]] < (3 if p[1] in three_people_events else 2)
                ):
                    person_count[p[0]] += 1
                    event_count[p[1]] += 1
                    z[i] = 1

        return Z


print("Defining Problem")
problem = MyProblem()

print("Initializing Algorithm")
algorithm = GA(
    pop_size=500,
    sampling=MySampling(),
    # sampling=BinaryRandomSampling(),
    crossover=BinaryCrossover(),
    mutation=MyMutation(),
    repair=RepairEvents(),
    eliminate_duplicates=True,
)

print("Optimizing...")
res = minimize(
    problem,
    algorithm,
    # termination=("time", "08:00:00"),
    return_least_infeasible=True,
    verbose=True,
)

# print("Best solution found: %s" % res.X.astype(int))
print("Found Solution!")
solution = res.X.astype(int)
print("Function value: %s" % res.F)
print("Constraint violation: %s" % res.CV)

solution_pairs = []
for i, p in enumerate(pairs):
    if solution[i] == 1:
        solution_pairs.append(p)

sol_people: list[str] = list(sorted(set(x[0] for x in solution_pairs)))
sol_events: list[str] = list(sorted(set(x[1] for x in solution_pairs)))
header = ["Name", "Count"] + sol_events
people_out = {p: [""] * (len(sol_events)) for p in sol_people}
counts = [0] * len(sol_events)
for person, event, _ in solution_pairs:
    index = sol_events.index(event)
    people_out[person][index] = "X"
    counts[index] += 1

with open("output.csv", "w") as f:
    f.write(",".join(header) + "\n")
    for person, out in people_out.items():
        f.write(",".join([person, str(sum(1 for x in out if x != ""))] + out) + "\n")
    f.write(",".join(["", ""] + list(map(str, counts))) + "\n")
