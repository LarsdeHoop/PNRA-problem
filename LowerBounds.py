from InstanceClass import Instance
from ILP_Formulation import ILP_formulation
import datetime
import gurobipy as gp
from gurobipy import GRB
from HelperFunctions import *

def compute_continuity_lb(instance):
    """Uses a set cover formulation to compute a lower bound for the continuity of care objective"""
    start_time = datetime.datetime.now()

    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    model = gp.Model(f"Continuity lower bound {instance.instance_name}", env=env)
    model.Params.LogToConsole = False

    ever_assigned = model.addVars(instance.nurse_ids, instance.patient_ids, lb=0.0, ub=1.0, vtype=GRB.BINARY,
                                  name="ever_assigned")

    model.addConstrs(gp.quicksum(ever_assigned[n, p] for n in instance.nurse_ids
                                 if s in instance.nurses[n]["shifts"]) >= 1
                     for p in instance.patient_ids for s in instance.patients[p]["shifts"])

    model.setObjective(gp.quicksum(ever_assigned[n, p] for n in instance.nurse_ids for p in instance.patient_ids),
                       GRB.MINIMIZE)

    model.optimize()
    end_time = datetime.datetime.now()
    time_taken = end_time - start_time
    return model.ObjVal, time_taken

def compute_gender_lb(instance):
    """Uses a simplified ILP to compute a lower bound for the gender requirements objective"""
    start_time = datetime.datetime.now()

    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    model = gp.Model(f"Gender-Mixing lower bound {instance.instance_name}", env=env)
    model.Params.LogToConsole = False

    # Variable for patient-to-room assignment
    valid_y_indices = [(p, r) for p in instance.patient_ids for r in instance.room_ids]
    y = model.addVars(valid_y_indices, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="y")

    ## Patient-to-Room assignment
    # Each patient is only assigned to one room
    model.addConstrs(gp.quicksum(y[p, r] for r in instance.room_ids) == 1 for p in instance.patient_ids)

    # Each room cannot exceed its capacity
    model.addConstrs(gp.quicksum(y[p, r] for p in instance.patients_per_shift[s])
                     <= instance.room_capacities[r] for r in instance.room_ids for s in instance.early_shifts)

    # Patients cannot be assigned to incompatible rooms
    model.addConstrs(
        y[p, r] == 0 for p in instance.patient_ids for r in instance.patients[p]["incompatible_rooms"])

    # Keep current occupants in the same room as before
    model.addConstrs(y[o, instance.patients[o]["prev_room"]] == 1 for o in instance.occupant_ids)

    # Variable for gender mixing constraint
    f_in_room = model.addVars(instance.room_ids, instance.early_shifts, lb=0.0, ub=1.0, vtype=GRB.BINARY,
                              name="f_in_room")
    m_in_room = model.addVars(instance.room_ids, instance.early_shifts, lb=0.0, ub=1.0, vtype=GRB.BINARY,
                              name="m_in_room")
    gender_vio = model.addVars(instance.room_ids, instance.early_shifts, lb=0.0, ub=1.0, vtype=GRB.BINARY,
                               name="gender_vio")

    # set m_in_room and f_in_room correctly based on assigned patients
    model.addConstrs(y[p, r] <= m_in_room[r, s] for s in instance.early_shifts for p in instance.patients_per_shift[s]
                     for r in instance.room_ids if instance.patients[p]["gender"] == "A")
    model.addConstrs(y[p, r] <= f_in_room[r, s] for s in instance.early_shifts for p in instance.patients_per_shift[s]
                     for r in instance.room_ids if instance.patients[p]["gender"] == "B")

    # If both male and female patients, add a violation
    model.addConstrs(m_in_room[r, s] + f_in_room[r, s] <= 1 + gender_vio[r, s] for r in instance.room_ids
                     for s in instance.early_shifts)

    model.setObjective(gp.quicksum(gender_vio[r, s] for r in instance.room_ids for s in instance.early_shifts),
                       GRB.MINIMIZE)

    model.optimize()
    end_time = datetime.datetime.now()
    time_taken = end_time - start_time
    return model.ObjVal, time_taken

def compute_skill_lb(instance):
    """Assign each patient to the highest skilled nurse to get a lower bound for the skill requirement objective"""
    start_time = datetime.datetime.now()
    skill_lb = 0
    for s in instance.all_shifts:
        nurses_present = instance.nurses_per_shift[s]

        # determine highest skill of the nurses present
        max_skill = 0
        for n in nurses_present:
            nurse_skill = instance.nurses[n]["skill"]
            if nurse_skill > max_skill:
                max_skill = nurse_skill
        dprint(f"In shift {s}: Highest skill level is {max_skill}")

        for p in instance.patients_per_shift[s]:
            skill_req = instance.patients[p]["skill"][s]
            dprint(f"\tSkill req {skill_req}")
            if skill_req > max_skill:
                skill_lb += skill_req - max_skill
                dprint(f"\t\t skill_lb += {skill_req - max_skill}")

    end_time = datetime.datetime.now()
    time_taken = end_time - start_time
    return skill_lb, time_taken

def compute_workload_lb(instance):
    """
    Uses a direct nurse-to-patient ILP to compute a lower bound for the workload violation objective.
    Each shift is optimized separately.
    """
    start_time = datetime.datetime.now()
    workload_lb = 0

    for s in instance.all_shifts:
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()
        model = gp.Model(f"Workload lower bound {instance.instance_name}", env=env)
        model.Params.LogToConsole = False

        valid_z_indices = [(n, p, s) for n in instance.nurses_per_shift[s] for p in instance.patients_per_shift[s]]
        z = model.addVars(valid_z_indices, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="z")

        valid_load_indices = [(n, s) for n in instance.nurses_per_shift[s]]
        load_vio = model.addVars(valid_load_indices, lb=0.0, vtype=GRB.CONTINUOUS, name="load_vio")

        model.addConstrs(gp.quicksum(z[n, p, s] for n in instance.nurses_per_shift[s])
                         == 1 for p in instance.patients_per_shift[s])

        model.addConstrs(gp.quicksum(instance.patients[p]["workload"][s] * z[n, p, s]
                                     for p in instance.patients_per_shift[s])
                         <= instance.nurses[n]["max_load"][s] + load_vio[n, s]
                         for n in instance.nurses_per_shift[s])

        model.setObjective(gp.quicksum(load_vio[n, s] for (n, s) in valid_load_indices), GRB.MINIMIZE)

        model.optimize()

        workload_lb += model.ObjVal

    end_time = datetime.datetime.now()
    time_taken = end_time - start_time
    return workload_lb, time_taken

def compute_imbalance_lb(instance):
    """
    Uses a direct nurse-to-patient ILP to compute a lower bound for the workload imbalance objective.
    Each shift is optimized separately.
    """
    start_time = datetime.datetime.now()
    imbalance_lb = 0
    instance_size = get_instance_size(instance)

    for s in instance.all_shifts:
        # print(f"Shift {s}")
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()
        model = gp.Model(f"Workload lower bound {instance.instance_name}", env=env)
        model.Params.TimeLimit = 300
        model.Params.LogToConsole = False

        valid_z_indices = [(n, p, s) for n in instance.nurses_per_shift[s] for p in instance.patients_per_shift[s]]
        z = model.addVars(valid_z_indices, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="z")

        min_load = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="min_load")
        max_load = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="max_load")

        model.addConstrs(gp.quicksum(z[n, p, s] for n in instance.nurses_per_shift[s])
                         == 1 for p in instance.patients_per_shift[s])

        model.addConstrs(min_load <= gp.quicksum(instance.patients[p]["workload"][s]
                                                 / instance.nurses[n]["max_load"][s] * z[n, p, s]
                                                 for p in instance.patients_per_shift[s])
                         for n in instance.nurses_per_shift[s])
        model.addConstrs(max_load >= gp.quicksum(instance.patients[p]["workload"][s]
                                                 / instance.nurses[n]["max_load"][s] * z[n, p, s]
                                                 for p in instance.patients_per_shift[s])
                         for n in instance.nurses_per_shift[s])

        model.setObjective(max_load - min_load, GRB.MINIMIZE)
        model.optimize()

        if model.Status == GRB.Status.TIME_LIMIT:
            print(f"\tTime limit reached. Suboptimal solution (obj {model.ObjVal:.3f}, bound {model.ObjBound:.3f}, gap {100 * model.MIPGap:.2f}%)")

        imbalance_lb += model.ObjBound

    end_time = datetime.datetime.now()
    time_taken = end_time - start_time
    return imbalance_lb, time_taken

def compute_full_lb_partial(instance, time_limit = 120, print_ilp_log=False):
    """
    Uses a direct nurse-to-patient ILP to compute a lower bound for the combined objective function.
    Since the gender-mixing objective is separate from the NPA, we calculate this lower bound separately.
    """
    gender_lb, gender_time = compute_gender_lb(instance)
    start_time = datetime.datetime.now()

    env = gp.Env(empty=True)
    if not print_ilp_log:
        env.setParam('OutputFlag', 0)
    env.start()

    model = gp.Model(instance.instance_name, env=env)
    model.Params.TimeLimit = time_limit
    model.Params.Method = 2 # determined work best
    if not print_ilp_log:
        model.Params.LogToConsole = False

    #################
    #   VARIABLES   #
    #################

    # Variable for nurse-to-patient assignment
    valid_z_indices = [(n, p, s) for n in instance.nurse_ids for p in instance.patient_ids
                       for s in instance.nurses[n]["shifts"] if s in instance.patients[p]["shifts"]]
    z = model.addVars(valid_z_indices, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="z")

    #################
    #  Constraints  #
    #################

    # Only one nurse assigned to each patient in each shift
    model.addConstrs(gp.quicksum(z[n, p, s] for n in instance.nurses_per_shift[s])
                     == 1 for s in instance.all_shifts for p in instance.patients_per_shift[s])

    #################
    #   OBJECTIVE   #
    #################

    obj = gp.LinExpr()

    ## Minimizing number of different nurses
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
        obj += instance.weights["Continuity"] * gp.quicksum(
            ever_assigned[n, p] for n in instance.nurse_ids for p in instance.patient_ids)

    ## Skill requirement
    if instance.weights["Skill Requirements"] != 0:
        valid_indices = [(p, s) for p in instance.patient_ids for s in instance.patients[p]["shifts"]]
        skill_vio = model.addVars(valid_indices, lb=0.0, vtype=GRB.CONTINUOUS)

        model.addConstrs(skill_vio[p, s] >= instance.patients[p]["skill"][s] -
                         gp.quicksum(instance.nurses[n]["skill"] * z[n, p, s]
                                     for n in instance.nurses_per_shift[s])
                         for p in instance.patient_ids for s in instance.patients[p]["shifts"])

        obj += instance.weights["Skill Requirements"] * gp.quicksum(skill_vio[p, s] for p in instance.patient_ids for s in instance.patients[p]["shifts"])

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

    # Minimizing workload imbalance per shift
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
        obj += instance.weights["Workload Imbalance"] * gp.quicksum(max_load[s] - min_load[s] for s in instance.all_shifts)

    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    if model.Status == GRB.Status.TIME_LIMIT:
        print(f"\tTime limit reached. Suboptimal solution (obj {model.ObjVal:.3f}, bound {model.ObjBound:.3f}, gap {100*model.MIPGap:.2f}%)")

    total_lb = instance.weights["Gender-Mixing"] * gender_lb + model.ObjBound
    end_time = datetime.datetime.now()
    time_taken = end_time - start_time + gender_time
    return total_lb, time_taken

def compute_lb_linear(instance, lb_to_calculate = "Full"):
    weights_copy = instance.weights.copy()

    if lb_to_calculate != "Full":
        instance.weights = {"Continuity": 0, "Gender-Mixing": 0, "Skill Requirements": 0,
                            "Workload Violation": 0, "Workload Imbalance": 0}
        if lb_to_calculate == "Continuity":
            instance.weights["Continuity"] = 1
        elif lb_to_calculate == "Gender-Mixing":
            instance.weights["Gender-Mixing"] = 1
        elif lb_to_calculate == "Skill Requirements":
            instance.weights["Skill Requirements"] = 1
        elif lb_to_calculate == "Workload Violation":
            instance.weights["Workload Violation"] = 1
        elif lb_to_calculate == "Workload Imbalance":
            instance.weights["Workload Imbalance"] = 1
        else:
            raise Exception(f"Invalid value for lb_to_calculate {lb_to_calculate}")

    start_time = datetime.datetime.now()
    model_information, solution_information, solution = ILP_formulation(instance,relax_problem = True, print_ilp_log= False)
    lower_bound = solution_information[0]

    end_time = datetime.datetime.now()
    time_taken = end_time - start_time

    instance.weights = weights_copy # reset the weights to what they were before
    return lower_bound, time_taken

if __name__ == "__main__":
    ##########################
    # Initialize instance
    instance_name = "test01"
    dataset_name = instance_name_to_dataset(instance_name)
    file_path = "Instances/" + dataset_name + "/" + instance_name + ".json"
    instance = Instance(file_path, print_instance_info=False)


    ##########################
    # Calculate the lower bound for each objective component (partial relaxation)
    print("Computing partial relaxation lower bounds:")
    continuity_lb, continuity_time = compute_continuity_lb(instance)
    print(f"\tContinuity of care: {continuity_lb:.3f}. Computation time = {continuity_time}")
    gender_lb, gender_time = compute_gender_lb(instance)
    print(f"\tGender-mixing: {gender_lb:.0f}. Computation time = {gender_time}")
    skill_lb, skill_time = compute_skill_lb(instance)
    print(f"\tSkill violations: {skill_lb}. Computation time = {skill_time}")
    workload_lb, workload_time = compute_workload_lb(instance)
    print(f"\tWorkload violations: {workload_lb:.0f}. Computation time = {workload_time}")
    imbalance_lb, imbalance_time = compute_imbalance_lb(instance)
    print(f"\tWorkload imbalance: {imbalance_lb:.3f}. Computation time = {imbalance_time}")
    full_lb, full_time= compute_full_lb_partial(instance, time_limit = 120, print_ilp_log=False)
    print(f"\tFull objective: {full_lb:.3f}. Computation time = {full_time}")

    ##########################
    # Calculate the lower bound for each objective component (linear relaxation)
    print("\nComputing LP relaxation lower bounds:")
    continuity_lb, continuity_time = compute_lb_linear(instance, "Continuity")
    print(f"\tContinuity of care: {continuity_lb}.  Computation time = {continuity_time}")
    gender_lb, gender_time = compute_lb_linear(instance, "Gender-Mixing")
    print(f"\tGender-mixing: {gender_lb}. Computation time = {gender_time}")
    skill_lb, skill_time = compute_lb_linear(instance, "Skill Requirements")
    print(f"\tSkill violations: {skill_lb}. Computation time = {skill_time}")
    workload_lb, workload_time = compute_lb_linear(instance, "Workload Violation")
    print(f"\tWorkload violations: {workload_lb}. Computation time = {workload_time}")
    imbalance_lb, imbalance_time = compute_lb_linear(instance, "Workload Imbalance")
    print(f"\tWorkload imbalance: {imbalance_lb:.3f}. Computation time = {imbalance_time}")
    full_lb, full_time = compute_lb_linear(instance, "Full")
    print(f"\tFull objective: {full_lb:.3f}. Computation time = {full_time}")
