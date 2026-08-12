from ComputeObjective import print_obj_table

def compute_objective_emergency(instance, solution, print_table = False):
    # solution might not fully have scheduled all patients
    obj_weights = instance.weights

    columns = [[],[]]
    columns[0] = ["Continuity of care", "Gender-Mixing", "Skill violations",
                "Workload violations", "Relative workload imbalance", "Total"]

    cont_of_care = compute_continuity_of_care(instance, solution)
    gender_vio = compute_gender_violation(instance, solution)
    skill_violations = compute_skill_violation(instance, solution)
    workload_violations = compute_workload_violation(instance,  solution)
    workload_imbalance = compute_workload_imbalance(instance, solution)
    numb_transfers = compute_numb_transfers(instance,solution)

    obj_values = [cont_of_care, gender_vio, skill_violations, workload_violations, workload_imbalance, numb_transfers]
    obj_components = {"Continuity": cont_of_care,
                      "Gender-Mixing": gender_vio,
                      "Skill Requirements": skill_violations,
                      "Workload Violation": workload_violations,
                      "Workload Imbalance": workload_imbalance,
                      "Transfers": numb_transfers}

    total_obj = obj_weights["Continuity"] * cont_of_care + \
                obj_weights["Gender-Mixing"] * gender_vio + \
                obj_weights["Skill Requirements"] * skill_violations + \
                obj_weights["Workload Violation"] * workload_violations + \
                obj_weights["Workload Imbalance"] * workload_imbalance+ \
                obj_weights["Transfers"] * numb_transfers

    if print_table:
        obj_weights_list = [obj_weights["Continuity"], obj_weights["Gender-Mixing"], obj_weights["Skill Requirements"],
                            obj_weights["Workload Violation"], obj_weights["Workload Imbalance"], obj_weights["Transfers"]]
        obj_names = ["Continuity of care", "Gender-Mixing", "Skill violations", "Workload violations",
                     "Relative workload imbalance", "Numb. transfers"]
        print_obj_table(obj_weights_list, obj_values, total_obj, obj_names)

    return total_obj, obj_components

def compute_continuity_of_care(instance, solution):
    (PR_assignment, NR_assignment) = solution

    nurses_used_per_patient = dict()
    for (p, s) in PR_assignment:
        if p not in nurses_used_per_patient:
            nurses_used_per_patient[p] = []
        r = PR_assignment[p,s]
        for i in range(3):
            for n in instance.nurses_per_shift[s+i]:
                if r in NR_assignment[n,s+i]:
                    if n not in nurses_used_per_patient[p]:
                        nurses_used_per_patient[p].append(n)

    CoC = 0
    for p in nurses_used_per_patient:
        CoC += len(nurses_used_per_patient[p])
    return CoC

def compute_gender_violation(instance, solution):
    (PR_assignment, NR_assignment) = solution

    gender_per_room = {(r,s):"Empty" for r in instance.room_ids for s in instance.early_shifts}
    for (p,s) in PR_assignment:
        if s in instance.early_shifts:
            r = PR_assignment[p,s]
            gender = instance.patients[p]["gender"]
            if gender_per_room[r,s] != gender:
                if gender_per_room[r,s] == "Empty":
                    gender_per_room[r, s] = gender
                else:
                    gender_per_room[r, s] = "Both"

    gender_vio = 0
    for (r,s) in gender_per_room.keys():
        if gender_per_room[r,s] == "Both":
            gender_vio += 1
    return gender_vio

def compute_skill_violation(instance, solution):
    (PR_assignment, NR_assignment) = solution

    skill_violations = 0
    for (p,s) in PR_assignment:
        r = PR_assignment[p, s]
        for i in range(3):
            skill_required = instance.patients[p]["skill"][s+i]
            for n in instance.nurses_per_shift[s+i]:
                if r in NR_assignment[n, s+i]:
                    nurse_skill = instance.nurses[n]["skill"]
                    if nurse_skill < skill_required:
                        skill_violations += skill_required - nurse_skill
    return skill_violations

def compute_workload_violation(instance, solution):
    (PR_assignment, NR_assignment) = solution

    workload_per_room = {(r,s): 0 for r in instance.room_ids for s in instance.all_shifts}
    for (p, s) in PR_assignment:
        r = PR_assignment[p,s]
        for i in range(3):
            load = instance.patients[p]["workload"][s+i]
            workload_per_room[r,s+i] += load

    workload_violations = 0
    for n in instance.nurse_ids:
        for s in instance.nurses[n]["shifts"]:
            workload = 0
            for r in NR_assignment[n,s]:
                workload += workload_per_room[r,s]
            if workload > instance.nurses[n]["max_load"][s]:
                workload_violations += workload - instance.nurses[n]["max_load"][s]
    return workload_violations

def compute_workload_imbalance(instance, solution):
    (PR_assignment, NR_assignment) = solution

    workload_per_room = {(r,s): 0 for r in instance.room_ids for s in instance.all_shifts}
    for (p, s) in PR_assignment:
        r = PR_assignment[p,s]
        for i in range(3):
            load = instance.patients[p]["workload"][s+i]
            workload_per_room[r,s+i] += load

    workload_imbalance = 0
    for s in instance.all_shifts:
        max_load = 0
        min_load = float("inf")
        for n in instance.nurses_per_shift[s]:
            workload = 0
            for r in NR_assignment[n, s]:
                workload += workload_per_room[r, s]
            rel_workload = workload / instance.nurses[n]["max_load"][s]
            if rel_workload > max_load:
                max_load = rel_workload
            if rel_workload < min_load:
                min_load = rel_workload
        workload_imbalance += max_load - min_load

    return workload_imbalance

def compute_numb_transfers(instance, solution):
    (PR_assignment, NR_assignment) = solution

    numb_transfers = 0
    for (p,s) in PR_assignment:
        if s in instance.early_shifts:
            adm_shift = instance.patients[p]["shifts"][0]
            if s != adm_shift:
                cur_room = PR_assignment[p, s]
                prev_room = PR_assignment[p, s - 3]
                if cur_room != prev_room:
                    numb_transfers += 1

    for o in instance.occupant_ids:
        if PR_assignment[o,0] != instance.patients[o]["prev_room"]:
            numb_transfers += 1

    return numb_transfers

################
# LOCAL SEARCH #
################

def compute_objective_emergency_LS(instance, solution, print_table = False):
    cap_penalty_weight = 100 * max(instance.weights.values())

    # for simulated annealing, solution is the global solution, not the local one
    regular_obj, obj_components = compute_objective_emergency(instance, solution)

    cap_penalty = get_capacity_penalty(instance,  solution)
    obj_components["Cap_penalty"] = cap_penalty

    total_obj = regular_obj + cap_penalty * cap_penalty_weight

    if print_table:
        obj_weights = instance.weights
        obj_weights_list = [obj_weights["Continuity"], obj_weights["Gender-Mixing"], obj_weights["Skill Requirements"],
                            obj_weights["Workload Violation"], obj_weights["Workload Imbalance"], obj_weights["Transfers"],
                            cap_penalty_weight]

        obj_values = [obj_components["Continuity"], obj_components["Gender-Mixing"], obj_components["Skill Requirements"],
                      obj_components["Workload Violation"], obj_components["Workload Imbalance"], obj_components["Transfers"],
                      obj_components["Cap_penalty"]]
        obj_names = ["Continuity of care", "Gender-mixing", "Skill violations", "Workload violations",
                     "Relative workload imbalance", "Transfers", "Capacity penalty"]
        print_obj_table(obj_weights_list, obj_values, total_obj, obj_names)


    return total_obj, obj_components


def get_capacity_penalty(instance, solution):
    (PR_assignment, NR_assignment) = solution

    numb_of_patients_per_room = dict()
    for (p,s) in PR_assignment:
        if s in instance.early_shifts:
            r = PR_assignment[p,s]
            if (r,s) in numb_of_patients_per_room:
                numb_of_patients_per_room[(r,s)] += 1
            else:
                numb_of_patients_per_room[(r,s)] = 1

    capacity_penalty = 0
    for (r,s) in numb_of_patients_per_room.keys():
        numb_patients = numb_of_patients_per_room[(r,s)]
        room_cap = instance.room_capacities[r]
        if numb_patients > room_cap:
            diff = numb_patients - room_cap
            capacity_penalty += diff
    return capacity_penalty