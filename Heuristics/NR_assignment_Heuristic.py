from HelperFunctions import *
import gurobipy as gp
from gurobipy import GRB
import random

def create_NR_assignment_sequential(instance, PR_assignment, shift_order = "chrono", sign=1, print_ilp_log = False):
    """Sequential nurse assignment"""
    NR_assignment = {(n, s): [] for n in instance.nurse_ids for s in instance.nurses[n]["shifts"]}
    nurses_per_patient = {p:[] for p in instance.patient_ids}

    all_shifts_sorted = instance.all_shifts.copy()
    if shift_order == "chrono":
        all_shifts_sorted = sorted(all_shifts_sorted, key=lambda s: sign * s)
    elif shift_order == "nurses":
        all_shifts_sorted = sorted(all_shifts_sorted, key=lambda s: sign * len(instance.nurses_per_shift[s]))
    elif shift_order == "patients":
        all_shifts_sorted = sorted(all_shifts_sorted, key=lambda s: sign * len(instance.patients_per_shift[s]))
    elif shift_order == "workload":
        workload_per_shift = {s: sum([instance.patients[p]["workload"][s] for p in instance.patients_per_shift[s]])
                              for s in instance.all_shifts}
        all_shifts_sorted = sorted(all_shifts_sorted, key=lambda s: sign * workload_per_shift[s])
    elif shift_order == "avg_workload":
        workload_per_shift = {s: sum([instance.patients[p]["workload"][s] for p in instance.patients_per_shift[s]])
                              for s in instance.all_shifts}
        numb_patients_per_shift = {s: len(instance.patients_per_shift[s]) for s in instance.all_shifts}
        for s in instance.all_shifts:
            if numb_patients_per_shift[s] == 0:
                numb_patients_per_shift[s] = 1  # to prevent div by zero error
        all_shifts_sorted = sorted(all_shifts_sorted,
                                      key=lambda s: sign * (workload_per_shift[s] / numb_patients_per_shift[s]))
    else:
        random.shuffle(all_shifts_sorted)

    for s in all_shifts_sorted:
        nurses_present = instance.nurses_per_shift[s]
        patients_present = instance.patients_per_shift[s]

        env = gp.Env(empty=True)
        if not print_ilp_log:
            env.setParam('OutputFlag', 0)
        env.start()

        model = gp.Model(f"PR-assignment {instance.instance_name} - shift {s}", env=env)

        if not print_ilp_log:
            model.Params.LogToConsole = False
        model.Params.TimeLimit = 30
        model.Params.Seed = random.randint(1, 100000)

        # Variable for nurse-to-room assignment
        valid_x_indices = [(n, r) for n in nurses_present for r in instance.room_ids]
        x = model.addVars(valid_x_indices, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="x")

        # Variable for nurse-to-patient assignment
        valid_z_indices = [(n, p) for n in nurses_present for p in patients_present]
        z = model.addVars(valid_z_indices, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="z")

        # Each room assigned to exactly one nurse
        model.addConstrs(gp.quicksum(x[n, r] for n in nurses_present) == 1 for r in instance.room_ids)

        model.addConstrs(z[n, p] == x[n, PR_assignment[p]] for n in nurses_present for p in patients_present)

        obj = gp.LinExpr()

        ## Minimizing number of different nurses
        if instance.weights["Continuity"] != 0:
            ever_assigned = model.addVars(nurses_present, patients_present, lb=0.0, ub=1.0, vtype=GRB.BINARY,
                                          name="ever_assigned")

            # set the previously assigned nurses to 1, otherwise depends on z_np
            for p in patients_present:
                for n in nurses_present:
                    if n in nurses_per_patient[p]:
                        model.addConstr(ever_assigned[n,p] == 1)
                    else:
                        model.addConstr(ever_assigned[n,p] == z[n,p])

            # Add to objective
            obj += instance.weights["Continuity"] * gp.quicksum(
                ever_assigned[n, p] for n in nurses_present for p in patients_present)

        ## Skill requirement
        if instance.weights["Skill Requirements"] != 0:
            skill_vio = model.addVars(patients_present, lb=0.0, vtype=GRB.CONTINUOUS)

            model.addConstrs(skill_vio[p] >= instance.patients[p]["skill"][s] -
                             gp.quicksum(instance.nurses[n]["skill"] * z[n, p]
                                         for n in nurses_present)
                             for p in patients_present)

            obj += instance.weights["Skill Requirements"] * gp.quicksum(skill_vio[p] for p in patients_present)

        ## Minimizing workload violation
        if instance.weights["Workload Violation"] != 0:
            load_vio = model.addVars(nurses_present, lb=0.0, vtype=GRB.CONTINUOUS, name="load_vio")

            model.addConstrs(gp.quicksum(instance.patients[p]["workload"][s] * z[n, p]
                                         for p in patients_present)
                             <= instance.nurses[n]["max_load"][s] + load_vio[n]
                             for n in nurses_present)

            # Add to objective
            obj += instance.weights["Workload Violation"] * gp.quicksum(load_vio[n] for n in nurses_present)

        # Minimizing workload imbalance per shift
        if instance.weights["Workload Imbalance"] != 0:
            min_load = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="min_load")
            max_load = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="max_load")

            model.addConstrs(min_load <= gp.quicksum(instance.patients[p]["workload"][s]
                                                        / instance.nurses[n]["max_load"][s] * z[n, p]
                                                        for p in patients_present)
                             for n in nurses_present)
            model.addConstrs(max_load >= gp.quicksum(instance.patients[p]["workload"][s]
                                                        / instance.nurses[n]["max_load"][s] * z[n, p]
                                                        for p in patients_present)
                             for n in nurses_present)

            # Add to objective
            obj += instance.weights["Workload Imbalance"] * (max_load - min_load)


        model.setObjective(obj, GRB.MINIMIZE)
        model.optimize()

        # Update NR assignment
        for n in nurses_present:
            for r in instance.room_ids:
                if abs(x[n, r].X - 1) < 10e-6:  # possible rounding error
                    NR_assignment[n, s].append(r)

        if instance.weights["Continuity"] != 0:
            for p in patients_present:
                for n in nurses_present:
                    if abs(ever_assigned[n,p].X - 1) < 10e-6:
                        if n not in nurses_per_patient[p]:
                            nurses_per_patient[p].append(n)

    return NR_assignment

def create_NR_assignment_full(instance, PR_assignment, time_limit = 300, noRel_heur_time = 0, value_to_improve = None, print_ilp_log = False):
    """Non-sequential nurse assignment"""


    env = gp.Env(empty=True)
    if not print_ilp_log:
        env.setParam('OutputFlag', 0)
    env.start()
    model = gp.Model(f"NR-assignment {instance.instance_name}", env=env)

    if not print_ilp_log:
        model.Params.LogToConsole = False

    model.Params.TimeLimit = time_limit
    model.Params.NoRelHeurTime = noRel_heur_time

    # Variable for nurse-to-room assignment
    valid_x_indices = [(n, r, s) for n in instance.nurse_ids for r in instance.room_ids
                       for s in instance.nurses[n]["shifts"]]
    x = model.addVars(valid_x_indices, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="x")

    # Variable for nurse-to-patient assignment
    valid_z_indices = [(n, p, s) for n in instance.nurse_ids for p in instance.patient_ids
                       for s in instance.nurses[n]["shifts"] if s in instance.patients[p]["shifts"]]
    z = model.addVars(valid_z_indices, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="z")

    # Each room assigned to exactly one nurse
    model.addConstrs(gp.quicksum(x[n, r, s] for n in instance.nurse_ids if s in instance.nurses[n]["shifts"]) == 1
                     for r in instance.room_ids for s in instance.all_shifts)

    model.addConstrs(
        z[n, p, s] == x[n, PR_assignment[p], s] for n in instance.nurse_ids for p in instance.patient_ids
        for s in instance.all_shifts
        if s in instance.nurses[n]["shifts"] if s in instance.patients[p]["shifts"])

    obj = gp.LinExpr()

    ## Continuity of care
    if instance.weights["Continuity"] != 0:
        ever_assigned = model.addVars(instance.nurse_ids, instance.patient_ids, lb=0.0, ub=1.0, vtype=GRB.BINARY,
                                      name="ever_assigned")

        # ever_assigned = 1 if at least one time nurse n assigned to patient p
        model.addConstrs(ever_assigned[n, p] >= z[n, p, s] for s in instance.all_shifts
                         for n in instance.nurses_per_shift[s]
                         for p in instance.patients_per_shift[s])

        # If z = 0 for all shifts, then ever_assigned = 0
        model.addConstrs(ever_assigned[n, p] <= gp.quicksum(z[n, p, s] for s in instance.all_shifts
                                                            if s in instance.patients[p]["shifts"]
                                                            if s in instance.nurses[n]["shifts"])
                         for n in instance.nurse_ids for p in instance.patient_ids)

        # Add to objective
        obj += instance.weights["Continuity"] * gp.quicksum(ever_assigned[n, p] for n in instance.nurse_ids for p in instance.patient_ids)

    ## Skill requirement
    if instance.weights["Skill Requirements"] != 0:
        valid_indices = [(p, s) for p in instance.patient_ids for s in instance.patients[p]["shifts"]]
        skill_vio = model.addVars(valid_indices, lb=0.0, vtype=GRB.CONTINUOUS)

        model.addConstrs(skill_vio[p, s] >= instance.patients[p]["skill"][s] -
                         gp.quicksum(instance.nurses[n]["skill"] * z[n, p, s]
                                     for n in instance.nurses_per_shift[s])
                         for p in instance.patient_ids for s in instance.patients[p]["shifts"])

        obj += instance.weights["Skill Requirements"] * gp.quicksum(
            skill_vio[p, s] for p in instance.patient_ids for s in instance.patients[p]["shifts"])

    ## Minimizing workload violation
    if instance.weights["Workload Violation"] != 0:
        valid_indices = [(n, s) for n in instance.nurse_ids for s in instance.nurses[n]["shifts"]]
        load_vio = model.addVars(valid_indices, lb=0.0, vtype=GRB.CONTINUOUS, name="load_vio")

        model.addConstrs(gp.quicksum(instance.patients[p]["workload"][s] * z[n, p, s]
                                     for p in instance.patients_per_shift[s])
                         <= instance.nurses[n]["max_load"][s] + load_vio[n, s]
                         for n in instance.nurse_ids for s in instance.nurses[n]["shifts"])

        # Add to objective
        obj += instance.weights["Workload Violation"] * gp.quicksum(
            load_vio[n, s] for n in instance.nurse_ids for s in instance.nurses[n]["shifts"])

    ## Minimizing workload imbalance per shift
    if instance.weights["Workload Imbalance"] != 0:
        min_load = model.addVars(instance.all_shifts, lb=0.0, vtype=GRB.CONTINUOUS, name="min_load")
        max_load = model.addVars(instance.all_shifts, lb=0.0, vtype=GRB.CONTINUOUS, name="max_load")

        model.addConstrs(min_load[s] <= gp.quicksum(instance.patients[p]["workload"][s]
                                                    / instance.nurses[n]["max_load"][s] * z[n, p, s]
                                                    for p in instance.patients_per_shift[s])
                         for n in instance.nurse_ids for s in instance.nurses[n]["shifts"])
        model.addConstrs(max_load[s] >= gp.quicksum(instance.patients[p]["workload"][s]
                                                    / instance.nurses[n]["max_load"][s] * z[n, p, s]
                                                    for p in instance.patients_per_shift[s])
                         for n in instance.nurse_ids for s in instance.nurses[n]["shifts"])

        # Add to objective
        obj += instance.weights["Workload Imbalance"] * gp.quicksum(
            max_load[s] - min_load[s] for s in instance.all_shifts)

    model.setObjective(obj, GRB.MINIMIZE)

    if value_to_improve is not None:
        class Callback:
            def __init__(self, model):
                self.improving_time = None

            def __call__(self, model, where):
                if where == GRB.Callback.MIP:
                    runtime = model.cbGet(GRB.Callback.RUNTIME)
                    incumbent = model.cbGet(GRB.Callback.MIP_OBJBST)
                    if incumbent < value_to_improve and self.improving_time is None:
                        print(f"Improved solution ({value_to_improve:.3f}) after {runtime} seconds")
                        self.improving_time = runtime
        callback = Callback(model)
        model.optimize(callback)
        time_until_better = callback.improving_time
    else:
        model.optimize()
        time_until_better = None

    if model.Status == GRB.Status.TIME_LIMIT:
        print(f"\tTime limit reached. Suboptimal solution (obj {model.ObjVal:.3f}, bound {model.ObjBound:.3f}, gap {100*model.MIPGap:.2f}%)")


    NR_assignment = {(n, s): [] for n in instance.nurse_ids for s in instance.nurses[n]["shifts"]}
    for n in instance.nurse_ids:
        for r in instance.room_ids:
            for s in instance.nurses[n]["shifts"]:
                if abs(x[n, r, s].X - 1) < 10e-6:  # possible rounding error
                    NR_assignment[n, s].append(r)

    return NR_assignment, time_until_better


if __name__ == "__main__":
    from InstanceClass import Instance
    from Heuristics.PR_assignment_Heuristic import create_PR_assignment
    from ComputeObjective import compute_objective

    random.seed(42)

    name = "test03"
    file_name = name + ".json"
    dataset_name = instance_name_to_dataset(name)
    file_path = "Instances/" + dataset_name + "/" + file_name
    instance = Instance(file_path, print_instance_info=True)

    PR_assignment = create_PR_assignment(instance)

    # do the nurse-to-patient assignment
    # NR_assignment = create_NR_assignment_greedy(instance, PR_assignment)
    NR_assignment = create_NR_assignment_sequential(instance, PR_assignment)
    # NR_assignment, time_until_better = create_NR_assignment_full(instance, PR_assignment)

    solution = (PR_assignment, NR_assignment)
    solution_attributes = get_solution_attributes(instance, solution)
    compute_objective(instance, solution_attributes, print_table = True)




























