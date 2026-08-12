from HelperFunctions import *

def compute_objective(instance, solution_attributes, print_table = False):
    obj_weights = instance.weights

    # calculate each objective component
    cont_of_care = compute_continuity_of_care(instance, solution_attributes)
    gender_vio = compute_gender_violation(instance, solution_attributes)
    skill_violations = compute_skill_violation(instance, solution_attributes)
    workload_violations = compute_workload_violation(instance,  solution_attributes)
    workload_imbalance = compute_workload_imbalance(instance, solution_attributes)

    obj_values = [cont_of_care, gender_vio, skill_violations, workload_violations, workload_imbalance]
    obj_components = {"Continuity": cont_of_care,
                      "Gender-Mixing": gender_vio,
                      "Skill Requirements": skill_violations,
                      "Workload Violation": workload_violations,
                      "Workload Imbalance": workload_imbalance}

    total_obj = obj_weights["Continuity"] * cont_of_care + \
                obj_weights["Gender-Mixing"] * gender_vio + \
                obj_weights["Skill Requirements"] * skill_violations + \
                obj_weights["Workload Violation"] * workload_violations + \
                obj_weights["Workload Imbalance"] * workload_imbalance

    if print_table:
        obj_weights_list = [obj_weights["Continuity"], obj_weights["Gender-Mixing"], obj_weights["Skill Requirements"],
                            obj_weights["Workload Violation"], obj_weights["Workload Imbalance"]]
        obj_names = ["Continuity of care", "Gender-Mixing", "Skill violations", "Workload violations", "Relative workload imbalance"]
        print_obj_table(obj_weights_list, obj_values, total_obj, obj_names)

    return total_obj, obj_components

def compute_continuity_of_care(instance, solution_attributes):
    nurse_count_per_patient = solution_attributes["nurse_count_per_patient"]

    CoC = 0
    for p in instance.patient_ids:
        CoC += len(nurse_count_per_patient[p])
    return CoC

def compute_gender_violation(instance, solution_attributes):
    gender_assignment = solution_attributes["gender_assignment"]
    gender_vio = 0
    for r in instance.room_ids:
        for s in instance.early_shifts:
            if gender_assignment[r,s] == "Both":
                gender_vio += 1
    return gender_vio

def compute_skill_violation(instance, solution_attributes):
    NP_assignment = solution_attributes["NP_assignment"]
    skill_violations = 0
    for p in instance.patient_ids:
        for s in instance.patients[p]["shifts"]:
            nurse_assigned = NP_assignment[p,s]
            skill_required = instance.patients[p]["skill"][s]
            nurse_skill = instance.nurses[nurse_assigned]["skill"]
            if nurse_skill < skill_required:
                skill_violations += skill_required - nurse_skill
    return skill_violations

def compute_workload_violation(instance, solution_attributes):
    workload_per_nurse = solution_attributes["workload_per_nurse"]

    workload_violations = 0
    for n in instance.nurse_ids:
        for s in instance.nurses[n]["shifts"]:
            workload = workload_per_nurse[n,s]
            if workload > instance.nurses[n]["max_load"][s]:
                workload_violations += workload - instance.nurses[n]["max_load"][s]
    return workload_violations

def compute_workload_imbalance(instance, solution_attributes):
    workload_per_nurse = solution_attributes["workload_per_nurse"]

    workload_imbalance = 0
    for s in instance.all_shifts:
        max_load = 0
        min_load = 10**10
        for n in instance.nurses_per_shift[s]:
            workload = workload_per_nurse[n,s]
            max_load = max([max_load, workload / instance.nurses[n]["max_load"][s]])
            min_load = min([min_load, workload / instance.nurses[n]["max_load"][s]])
        workload_imbalance += max_load - min_load
    return workload_imbalance

def print_obj_table(obj_weights,obj_values,total_obj, obj_names):
    columns = []
    alignments = []
    seperators = []
    columns_till_second_equality = 3
    int_obj_col = 3

    # first column
    columns.append(obj_names)
    alignments.append("<")
    seperators.append(" = ")

    # second column
    weight_strings = [str(w) for w in obj_weights]
    if max(["." in x for x in weight_strings]) == 0:  # if all weights are integer
        columns.append(weight_strings)
        alignments.append(">")
        seperators.append(" * ")
    else:
        int_part, dec_part = [], []
        for x in weight_strings:
            splitted = x.split(".")
            int_part.append(splitted[0])
            if len(splitted) > 1:
                dec_part.append("." + splitted[1])
            else:
                dec_part.append("")
        columns.append(int_part)
        columns.append(dec_part)
        alignments.append(">")
        alignments.append("<")
        seperators.append("")
        seperators.append(" * ")
        columns_till_second_equality += 1
        int_obj_col += 1

    # third column
    obj_strings = [fmt(x) for x in obj_values]
    if max(["." in x for x in obj_strings]) == 0:  # if all objective components are integer
        columns.append(obj_strings)
        alignments.append(">")
        seperators.append(" =  ") # give extra space for the total objective row
    else:
        int_part, dec_part = [], []
        for x in obj_strings:
            splitted = x.split(".")
            int_part.append(splitted[0])
            if len(splitted) > 1:
                dec_part.append("." + splitted[1])
            else:
                dec_part.append("")
        columns.append(int_part)
        columns.append(dec_part)
        alignments.append(">")
        alignments.append("<")
        seperators.append("")
        seperators.append(" =  ")  # give extra space for the total objective row
        columns_till_second_equality += 1
        int_obj_col += 1

    # fourth column
    full_obj_strings = [fmt(obj_weights[i] * obj_values[i]) for i in range(len(obj_values))]
    if max(["." in x for x in full_obj_strings]) == 0:  # if all objective components are integer
        columns.append(full_obj_strings)
        alignments.append(">")
    else:
        int_part, dec_part = [], []
        for x in full_obj_strings:
            splitted = x.split(".")
            int_part.append(splitted[0])
            if len(splitted) > 1:
                dec_part.append("." + splitted[1])
            else:
                dec_part.append("")
        columns.append(int_part)
        columns.append(dec_part)
        alignments.append(">")
        alignments.append("<")
        seperators.append("")

    column_widths = print_as_table(None, columns, align=alignments, sep=seperators)
    col_width_to_second_equality = sum([column_widths[i] for i in range(columns_till_second_equality)])
    sep_width_to_second_equality = sum([len(seperators[i]) for i in range(columns_till_second_equality - 1)])
    int_col_width = column_widths[int_obj_col]

    print(f"{str('Total objective'):<{col_width_to_second_equality + sep_width_to_second_equality}}", end=" = ")
    total_obj_str = fmt(total_obj)
    if "." not in total_obj_str:
        print(f"{total_obj_str:>{int_col_width + 1}}\n")
    else:
        split = total_obj_str.split(".")
        int_part = split[0]
        dec_part = split[1]
        print(f"{int_part:>{int_col_width + 1}}.{dec_part}\n")

def fmt(x, max_decimals = 3):
    # Round x and remove trailing decimals
    return f"{x:.{max_decimals}f}".rstrip("0").rstrip(".")

################
# LOCAL SEARCH #
################

def compute_objective_LS(instance, solution_attributes, print_table = False):
    cap_penalty_weight = 100 * max(instance.weights.values())

    # compute the regular objective value
    obj_value, obj_components = compute_objective(instance, solution_attributes,False)

    # compute capacity penalty and additional guide value
    cap_penalty = get_capacity_penalty(instance,  solution_attributes)

    # add them to the objective components
    obj_components["Cap_penalty"] = cap_penalty
    penalty_value = cap_penalty_weight * cap_penalty

    if cap_penalty == 0:
        feasible = True
    else:
        feasible = False

    if print_table:
        obj_weights = instance.weights
        obj_weights_list = [obj_weights["Continuity"], obj_weights["Gender-Mixing"], obj_weights["Skill Requirements"],
                            obj_weights["Workload Violation"], obj_weights["Workload Imbalance"],
                            cap_penalty_weight]

        obj_values = [obj_components["Continuity"], obj_components["Gender-Mixing"], obj_components["Skill Requirements"],
                      obj_components["Workload Violation"], obj_components["Workload Imbalance"],
                      obj_components["Cap_penalty"]]
        obj_names = ["Continuity of care", "Gender-mixing", "Skill violations", "Workload violations",
                     "Relative workload imbalance","Capacity penalty"]
        print_obj_table(obj_weights_list, obj_values, obj_value + penalty_value, obj_names)

    return obj_value, penalty_value, obj_components, feasible,

def get_capacity_penalty(instance, solution_attributes):
    patients_per_room = solution_attributes["patients_per_room"]

    capacity_penalty = 0
    for r in instance.room_ids:
        for s in instance.early_shifts:
            numb_patients_in_room = len(patients_per_room[r,s])
            over_capacity = numb_patients_in_room - instance.room_capacities[r]
            if over_capacity > 0:
                dprint(f"Room {r} is {over_capacity} patients over capacity during shift {s}")
                capacity_penalty += over_capacity
    dprint(f"Total capacity penalty = {capacity_penalty}")
    return capacity_penalty
