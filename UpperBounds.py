from InstanceClass import Instance
import datetime
import gurobipy as gp
from gurobipy import GRB
from HelperFunctions import *

def compute_continuity_ub(instance):
    """Uses a set cover formulation to compute an upper bound for the continuity of care objective"""
    start_time = datetime.datetime.now()

    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    model = gp.Model(f"Continuity upper bound {instance.instance_name}", env=env)
    model.Params.LogToConsole = False

    valid_z_indices = [(n, p, s) for n in instance.nurse_ids for p in instance.patient_ids
                       for s in instance.patients[p]["shifts"]
                       if s in instance.nurses[n]["shifts"]]
    z = model.addVars(valid_z_indices, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="z")

    ever_assigned = model.addVars(instance.nurse_ids, instance.patient_ids, lb=0.0, ub=1.0, vtype=GRB.BINARY,
                                  name="ever_assigned")

    model.addConstrs(gp.quicksum(z[n, p, s] for n in instance.nurse_ids if s in instance.nurses[n]["shifts"])
                     == 1 for p in instance.patient_ids for s in instance.patients[p]["shifts"])

    model.addConstrs(ever_assigned[n, p] >= z[n, p, s] for n in instance.nurse_ids for p in instance.patient_ids
                     for s in instance.patients[p]["shifts"] if s in instance.nurses[n]["shifts"])
    model.addConstrs(ever_assigned[n, p] <= gp.quicksum(z[n, p, s] for s in instance.patients[p]["shifts"] if
                                                        s in instance.nurses[n]["shifts"])
                     for n in instance.nurse_ids for p in instance.patient_ids)

    model.setObjective(gp.quicksum(ever_assigned[n, p] for n in instance.nurse_ids for p in instance.patient_ids),
                       GRB.MAXIMIZE)

    model.optimize()
    end_time = datetime.datetime.now()
    time_taken = end_time - start_time
    return model.ObjVal, time_taken

def compute_gender_ub(instance):
    """Uses a simplified ILP formulation to compute an upper bound for the gender requirements objective"""
    start_time = datetime.datetime.now()

    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    model = gp.Model(f"Gender-Mixing lower bound {instance.instance_name}", env=env)
    model.Params.LogToConsole = False

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

    ## ADDITIONAL CONSTRAINTS TO ENSURE CORRECT VALUES WHEN MAXIMIZING
    model.addConstrs(m_in_room[r,s] <= gp.quicksum(y[p, r] for p in instance.patients_per_shift[s]
                                                   if instance.patients[p]["gender"] == "A")
                        for s in instance.early_shifts for r in instance.room_ids)
    model.addConstrs(f_in_room[r, s] <= gp.quicksum(y[p, r] for p in instance.patients_per_shift[s]
                                                    if instance.patients[p]["gender"] == "B")
                     for s in instance.early_shifts for r in instance.room_ids)
    model.addConstrs(gender_vio[r, s] <= m_in_room[r, s] for r in instance.room_ids
                     for s in instance.early_shifts)
    model.addConstrs(gender_vio[r, s] <= f_in_room[r, s] for r in instance.room_ids
                     for s in instance.early_shifts)

    model.setObjective(gp.quicksum(gender_vio[r, s] for r in instance.room_ids for s in instance.early_shifts),
                       GRB.MAXIMIZE)

    model.optimize()
    end_time = datetime.datetime.now()
    time_taken = end_time - start_time
    return model.ObjVal, time_taken

def compute_skill_ub(instance):
    """Assign each patient to the lowest skilled nurse present to obtain an upper bound."""
    start_time = datetime.datetime.now()

    skill_violation = 0
    for s in instance.all_shifts:
        # determine lowest skill level in shift
        lowest_skill = instance.skillLevels
        for n in instance.nurses_per_shift[s]:
            nurse_skill = instance.nurses[n]["skill"]
            if nurse_skill < lowest_skill:
                lowest_skill = nurse_skill

        # assign all patients to that nurse
        for p in instance.patients_per_shift[s]:
            patient_skill_req = instance.patients[p]["skill"][s]
            if patient_skill_req > lowest_skill:
                skill_violation += patient_skill_req - lowest_skill


    end_time = datetime.datetime.now()
    time_taken = end_time - start_time
    return skill_violation, time_taken

def compute_workload_ub(instance):
    """Assign each patient to the nurse with the lowest max workload to obtain an upper bound."""
    start_time = datetime.datetime.now()

    workload_violation = 0
    for s in instance.all_shifts:
        # determine lowest max-load in shift
        lowest_maxload = float("inf")
        for n in instance.nurses_per_shift[s]:
            nurse_maxload = instance.nurses[n]["max_load"][s]
            if nurse_maxload < lowest_maxload:
                lowest_maxload = nurse_maxload

        # assign all patients to that nurse
        assigned_workload = 0
        for p in instance.patients_per_shift[s]:
            patient_workload = instance.patients[p]["workload"][s]
            assigned_workload += patient_workload

        if assigned_workload > lowest_maxload:
            workload_violation += assigned_workload - lowest_maxload

    end_time = datetime.datetime.now()
    time_taken = end_time - start_time
    return workload_violation, time_taken

def compute_imbalance_ub(instance):
    """Assign each patient to the nurse with the lowest max workload to obtain an upper bound."""
    start_time = datetime.datetime.now()

    workload_imbalance = 0
    for s in instance.all_shifts:
        # determine lowest max-load in shift
        if len(instance.nurses_per_shift[s]) > 0: # if there is only one nurse, there is no imbalance
            lowest_maxload = float("inf")
            for n in instance.nurses_per_shift[s]:
                nurse_maxload = instance.nurses[n]["max_load"][s]
                if nurse_maxload < lowest_maxload:
                    lowest_maxload = nurse_maxload

            # assign all patients to that nurse
            assigned_workload = 0
            for p in instance.patients_per_shift[s]:
                patient_workload = instance.patients[p]["workload"][s]
                assigned_workload += patient_workload

            workload_imbalance += assigned_workload / lowest_maxload

    end_time = datetime.datetime.now()
    time_taken = end_time - start_time
    return workload_imbalance, time_taken

def compute_full_ub_partial(instance, time_limit=120, print_ilp_log=False):
    """
    Uses a direct nurse-to-patient ILP to compute an upper bound for the combined objective function.
    Since the gender-mixing objective is separate from the NPA, we calculate this upper bound separately.
    """
    gender_ub, gender_time = compute_gender_ub(instance)
    start_time = datetime.datetime.now()

    env = gp.Env(empty=True)
    if not print_ilp_log:
        env.setParam('OutputFlag', 0)
    env.start()

    model = gp.Model(instance.instance_name, env=env)
    model.Params.TimeLimit = time_limit
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

    # ## Minimizing number of different nurses
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

        model.addConstrs(
            skill_vio[p, s] == gp.quicksum(max(0, instance.patients[p]["skill"][s] - instance.nurses[n]["skill"]) *
                                           z[n, p, s] for n in instance.nurses_per_shift[s])
            for p in instance.patient_ids for s in instance.patients[p]["shifts"])

        obj += instance.weights["Skill Requirements"] * gp.quicksum(
            skill_vio[p, s] for p in instance.patient_ids for s in instance.patients[p]["shifts"])

    ## Minimizing workload violation
    if instance.weights["Workload Violation"] != 0:
        valid_indices = [(n, s) for n in instance.nurse_ids for s in instance.nurses[n]["shifts"]]
        load_vio = model.addVars(valid_indices, lb=0.0, vtype=GRB.CONTINUOUS, name="load_vio")
        aux_var = model.addVars(valid_indices, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="aux_var")

        M = {(n, s): max(abs(sum(instance.patients[p]["workload"][s] for p in instance.patients_per_shift[s])
                             - instance.nurses[n]["max_load"][s]),
                         instance.nurses[n]["max_load"][s])
             for (n, s) in valid_indices}

        model.addConstrs(load_vio[n, s] <= gp.quicksum(instance.patients[p]["workload"][s] * z[n, p, s]
                                                       for p in instance.patients_per_shift[s])
                         - instance.nurses[n]["max_load"][s] + M[n, s] * (1 - aux_var[n, s])
                         for n in instance.nurse_ids for s in instance.nurses[n]["shifts"])

        model.addConstrs(load_vio[n, s] <= M[n, s] * aux_var[n, s]
                         for n in instance.nurse_ids for s in instance.nurses[n]["shifts"])

        model.addConstrs(gp.quicksum(load_vio[n, s] for n in instance.nurses_per_shift[s]) <=
                         max(0, sum(instance.patients[p]["workload"][s] for p in instance.patients_per_shift[s])
                             - min([instance.nurses[n]["max_load"][s] for n in instance.nurses_per_shift[s]]))
                         for s in instance.all_shifts)

        # Add to objective
        obj += instance.weights["Workload Violation"] * gp.quicksum(
            load_vio[n, s] for n in instance.nurse_ids for s in instance.nurses[n]["shifts"])

    ## Minimizing workload imbalance per shift
    if instance.weights["Workload Imbalance"] != 0:
        min_load = model.addVars(instance.all_shifts, lb=0.0, vtype=GRB.CONTINUOUS, name="min_load")
        max_load = model.addVars(instance.all_shifts, lb=0.0, vtype=GRB.CONTINUOUS, name="max_load")

        # Getting the max
        M = {s: sum(instance.patients[p]["workload"][s] for p in instance.patients_per_shift[s])
                / min(instance.nurses[n]["max_load"][s] for n in instance.nurses_per_shift[s])
             for s in instance.all_shifts}
        valid_indices = [(n, s) for n in instance.nurse_ids for s in instance.nurses[n]["shifts"]]
        aux_min_var = model.addVars(valid_indices, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="aux_min_var")

        model.addConstrs(min_load[s] >= gp.quicksum(instance.patients[p]["workload"][s]
                                                    / instance.nurses[n]["max_load"][s] * z[n, p, s]
                                                    for p in instance.patients_per_shift[s])
                         - (1 - aux_min_var[n, s]) * M[s]
                         for n in instance.nurse_ids for s in instance.nurses[n]["shifts"])
        model.addConstrs(gp.quicksum(aux_min_var[n, s] for n in instance.nurses_per_shift[s])
                         == 1
                         for s in instance.all_shifts)

        # Getting the max
        valid_indices = [(n, s) for n in instance.nurse_ids for s in instance.nurses[n]["shifts"]]
        aux_max_var = model.addVars(valid_indices, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="aux_max_var")

        model.addConstrs(max_load[s] <= gp.quicksum(instance.patients[p]["workload"][s]
                                                    / instance.nurses[n]["max_load"][s] * z[n, p, s]
                                                    for p in instance.patients_per_shift[s])
                         + (1 - aux_max_var[n, s]) * M[s]
                         for n in instance.nurse_ids for s in instance.nurses[n]["shifts"])
        model.addConstrs(gp.quicksum(aux_max_var[n, s] for n in instance.nurses_per_shift[s])
                         == 1
                         for s in instance.all_shifts)

        # Add to objective
        obj += instance.weights["Workload Imbalance"] * gp.quicksum(
            max_load[s] - min_load[s] for s in instance.all_shifts)

    model.setObjective(obj, GRB.MAXIMIZE)
    model.optimize()
    end_time = datetime.datetime.now()

    if model.status == GRB.OPTIMAL:
        total_ub = instance.weights["Gender-Mixing"] * gender_ub + model.ObjVal
    else:
        total_ub = instance.weights["Gender-Mixing"] * gender_ub + model.ObjBound
    time_taken = end_time - start_time + gender_time
    return total_ub, time_taken

def compute_full_ub_linear(instance, time_limit = None, print_ilp_log = False):
    """
    Uses the linearly relaxed ILP to compute an upper bound for the combined objective function.
    """
    start_time = datetime.datetime.now()

    env = gp.Env(empty=True)
    if not print_ilp_log:
        env.setParam('OutputFlag', 0)
    env.start()

    model = gp.Model(instance.instance_name, env=env)
    model.Params.Method = 2  # determined work best
    if not print_ilp_log:
        model.Params.LogToConsole = False

    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    #################
    #   VARIABLES   #
    #################

    # decision variable for nurses
    valid_x_indices = [(n, r, s) for n in instance.nurse_ids for r in instance.room_ids
                       for s in instance.nurses[n]["shifts"]]
    x = model.addVars(valid_x_indices, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="x")

    # decision variable for patients
    valid_y_indices = [(p, r) for p in instance.patient_ids for r in instance.room_ids]
    y = model.addVars(valid_y_indices, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="y")

    # Variable for nurse-to-patient assignment
    valid_z_indices = [(n, p, s) for n in instance.nurse_ids for p in instance.patient_ids
                       for s in instance.nurses[n]["shifts"] if s in instance.patients[p]["shifts"]]
    z = model.addVars(valid_z_indices, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="z")

    #################
    #  Constraints  #
    #################

    ## Nurse-to-Room assignment
    # Each room assigned to exactly one nurse
    model.addConstrs(gp.quicksum(x[n, r, s] for n in instance.nurses_per_shift[s]) == 1
                     for r in instance.room_ids for s in instance.all_shifts)

    ## Patient-to-Room assignment
    # Each patient is only assigned to one room
    model.addConstrs(gp.quicksum(y[p, r] for r in instance.room_ids) == 1 for p in instance.patient_ids)

    # Each room cannot exceed its capacity
    model.addConstrs(gp.quicksum(y[p, r] for p in instance.patients_per_shift[s])
                     <= instance.room_capacities[r] for r in instance.room_ids for s in instance.early_shifts)

    # Patients cannot be assigned to incompatible rooms
    model.addConstrs(y[p, r] == 0 for p in instance.patient_ids for r in instance.patients[p]["incompatible_rooms"])

    # Keep current occupants in the same room as before
    model.addConstrs(y[o, instance.patients[o]["prev_room"]] == 1 for o in instance.occupant_ids)

    ## Nurse-to-Patient assignment
    # Nurse is assigned to a patient if they are in the same room
    model.addConstrs(z[n, p, s] >= x[n, r, s] + y[p, r] - 1 for s in instance.all_shifts
                     for r in instance.room_ids for n in instance.nurses_per_shift[s]
                     for p in instance.patients_per_shift[s])

    # Only one nurse assigned to each patient in each shift
    model.addConstrs(gp.quicksum(z[n, p, s] for n in instance.nurses_per_shift[s])
                     == 1 for s in instance.all_shifts for p in instance.patients_per_shift[s])

    #################
    #   OBJECTIVE   #
    #################

    obj = gp.LinExpr()

    # ## Minimizing number of different nurses
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

    ## Minimizing number of gender violations
    if instance.weights["Gender-Mixing"] != 0:
        # Variable for gender mixing constraint
        f_in_room = model.addVars(instance.room_ids, instance.early_shifts, lb=0.0, ub=1.0, vtype=GRB.BINARY,
                                  name="f_in_room")
        m_in_room = model.addVars(instance.room_ids, instance.early_shifts, lb=0.0, ub=1.0, vtype=GRB.BINARY,
                                  name="m_in_room")
        gender_vio = model.addVars(instance.room_ids, instance.early_shifts, lb=0.0, ub=1.0, vtype=GRB.BINARY,
                                   name="gender_vio")

        # set m_in_room and f_in_room correctly based on assigned patients
        model.addConstrs(m_in_room[r, s] <= gp.quicksum(y[p,r] for p in instance.patients_per_shift[s]
                                                        if instance.patients[p]["gender"] == "A")
                         for r in instance.room_ids for s in instance.early_shifts)
        model.addConstrs(f_in_room[r, s] <= gp.quicksum(y[p,r] for p in instance.patients_per_shift[s]
                                                        if instance.patients[p]["gender"] == "B")
                         for r in instance.room_ids for s in instance.early_shifts)
        model.addConstrs(gender_vio[r, s] <= m_in_room[r, s]
                         for r in instance.room_ids for s in instance.early_shifts)
        model.addConstrs(gender_vio[r, s] <= f_in_room[r, s]
                         for r in instance.room_ids for s in instance.early_shifts)

        obj += instance.weights["Gender-Mixing"] * gp.quicksum(
            gender_vio[r, s] for r in instance.room_ids for s in instance.early_shifts)

    ## Skill requirement
    if instance.weights["Skill Requirements"] != 0:
        valid_indices = [(p, s) for p in instance.patient_ids for s in instance.patients[p]["shifts"]]
        skill_vio = model.addVars(valid_indices, lb=0.0, vtype=GRB.CONTINUOUS)

        model.addConstrs(skill_vio[p, s] <= gp.quicksum(max(0,instance.patients[p]["skill"][s] - instance.nurses[n]["skill"]) *
                                                        z[n,p,s] for n in instance.nurses_per_shift[s])
                         for p in instance.patient_ids for s in instance.patients[p]["shifts"])

        obj += instance.weights["Skill Requirements"] * gp.quicksum(
            skill_vio[p, s] for p in instance.patient_ids for s in instance.patients[p]["shifts"])

    ## Minimizing workload violation
    if instance.weights["Workload Violation"] != 0:
        valid_indices = [(n, s) for n in instance.nurse_ids for s in instance.nurses[n]["shifts"]]
        load_vio = model.addVars(valid_indices, lb=0.0, vtype=GRB.CONTINUOUS, name="load_vio")
        aux_var = model.addVars(valid_indices, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="aux_var")

        M = {(n, s):max(abs(sum(instance.patients[p]["workload"][s] for p in instance.patients_per_shift[s])
                    - instance.nurses[n]["max_load"][s]),
                        instance.nurses[n]["max_load"][s])
             for (n, s) in valid_indices}


        model.addConstrs(load_vio[n, s] <= gp.quicksum(instance.patients[p]["workload"][s] * z[n, p, s]
                                                        for p in instance.patients_per_shift[s])
                         - instance.nurses[n]["max_load"][s] + M[n,s] * (1-aux_var[n,s])
                         for n in instance.nurse_ids for s in instance.nurses[n]["shifts"])

        model.addConstrs(load_vio[n, s] <= M[n,s] * aux_var[n,s]
                         for n in instance.nurse_ids for s in instance.nurses[n]["shifts"])

        model.addConstrs(gp.quicksum(load_vio[n, s] for n in instance.nurses_per_shift[s]) <=
                         max(0, sum(instance.patients[p]["workload"][s] for p in instance.patients_per_shift[s])
                             - min([instance.nurses[n]["max_load"][s] for n in instance.nurses_per_shift[s]]))
                         for s in instance.all_shifts)

        # Add to objective
        obj += instance.weights["Workload Violation"] * gp.quicksum(
            load_vio[n, s] for n in instance.nurse_ids for s in instance.nurses[n]["shifts"])

    ## Minimizing workload imbalance per shift
    if instance.weights["Workload Imbalance"] != 0:
        min_load = model.addVars(instance.all_shifts, lb=0.0, vtype=GRB.CONTINUOUS, name="min_load")
        max_load = model.addVars(instance.all_shifts, lb=0.0, vtype=GRB.CONTINUOUS, name="max_load")

        # Getting the max
        M = {s: sum(instance.patients[p]["workload"][s] for p in instance.patients_per_shift[s])
                / min(instance.nurses[n]["max_load"][s] for n in instance.nurses_per_shift[s])
             for s in instance.all_shifts}
        valid_indices = [(n,s) for n in instance.nurse_ids for s in instance.nurses[n]["shifts"]]
        aux_min_var = model.addVars(valid_indices, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="aux_min_var")

        model.addConstrs(min_load[s] >= gp.quicksum(instance.patients[p]["workload"][s]
                                                    / instance.nurses[n]["max_load"][s] * z[n, p, s]
                                                    for p in instance.patients_per_shift[s])
                                        - (1-aux_min_var[n,s]) * M[s]
                         for n in instance.nurse_ids for s in instance.nurses[n]["shifts"])
        model.addConstrs(gp.quicksum(aux_min_var[n,s] for n in instance.nurses_per_shift[s])
                         == 1
                         for s in instance.all_shifts)

        # Getting the max
        valid_indices = [(n,s) for n in instance.nurse_ids for s in instance.nurses[n]["shifts"]]
        aux_max_var = model.addVars(valid_indices, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="aux_max_var")

        model.addConstrs(max_load[s] <= gp.quicksum(instance.patients[p]["workload"][s]
                                                    / instance.nurses[n]["max_load"][s] * z[n, p, s]
                                                    for p in instance.patients_per_shift[s])
                                        + (1-aux_max_var[n,s]) * M[s]
                         for n in instance.nurse_ids for s in instance.nurses[n]["shifts"])
        model.addConstrs(gp.quicksum(aux_max_var[n,s] for n in instance.nurses_per_shift[s])
                         == 1
                         for s in instance.all_shifts)

        # Add to objective
        obj += instance.weights["Workload Imbalance"] * gp.quicksum(
            max_load[s] - min_load[s] for s in instance.all_shifts)

    model.update()
    for v in model.getVars():
        v.setAttr('Vtype', 'CONTINUOUS')

    model.setObjective(obj, GRB.MAXIMIZE)
    model.optimize()
    end_time = datetime.datetime.now()

    return model.ObjVal, end_time - start_time

if __name__ == "__main__":
    ##########################
    # Initialize instance

    name = "test01"
    file_name = name + ".json"
    dataset_name = instance_name_to_dataset(name)
    file_path = "Instances/" + dataset_name + "/" + file_name
    instance = Instance(file_path, print_instance_info=False)

    ##########################
    # Calculate the upper bound for each objective component
    continuity_ub, continuity_time = compute_continuity_ub(instance)
    print(f"Continuity upper bound = {continuity_ub:.0f}. Computation time = {continuity_time}")
    gender_ub, gender_time = compute_gender_ub(instance)
    print(f"Gender-Mixing upper bound = {gender_ub:.0f}. Computation time = {gender_time}")
    skill_ub, skill_time = compute_skill_ub(instance)
    print(f"Skill requirement upper bound = {skill_ub}. Computation time = {skill_time}")
    workload_ub, workload_time = compute_workload_ub(instance)
    print(f"Workload upper bound = {workload_ub:.0f}. Computation time = {workload_time}")
    imbalance_ub, imbalance_time = compute_imbalance_ub(instance)
    print(f"Workload imbalance upper bound = {imbalance_ub:.3f}. Computation time = {imbalance_time}")
    relaxed_ub, relaxed_time = compute_full_ub_linear(instance, print_ilp_log=False)
    print(f"Full (linear relaxation) upper bound = {relaxed_ub:.3f}. Computation time = {relaxed_time}")
    partial_ub, partial_time = compute_full_ub_partial(instance, time_limit = 120, print_ilp_log=False)
    print(f"Full (partial relaxation) upper bound = {partial_ub:.3f}. Computation time = {partial_time}")

