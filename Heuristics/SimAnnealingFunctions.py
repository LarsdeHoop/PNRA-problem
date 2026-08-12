def get_SA_solution_attributes(instance, solution):
    """
    Get additional data structures to compute the change in objective when making a simulated annealing move.
    This is slightly different from the solution_attributes from HelperFunctions
    Will return a dictionary with the following keys:
        - cap_penalty_weight: weight used to calculate the capacity penalty. 100 times largest weight.
        - patients_per_room: a dictionary with for each room-shift pair a list of patients staying in that room
                             during that shift.
        - nurse_per_room: a dictionary with for each room-shift pair, the nurse assigned to that room.
        - gender_numb_per_room: a dictionary with for each room-shift-gender tuple the number of patients of
                                that gender staying in that room during that shift.
        - gender_assignment: a dictionary with for each room-early_shift pair the gender of the room on that day.
                             Value can be either "A", "B", "Both" or None.
        - NP_assignment: a dictionary with for each patient-shift pair, the nurse assigned to that patient
                         during that shift.
        - workload_per_room: a dictionary with for each room-shift pair, the assigned workload of that room
                             during that shift.
        - workload_per_nurse: a dictionary with for each nurse-shift pair, the assigned workload of that nurse
                              during that shift.
        - nurse_count_per_patient: a dictionary with for each patient, a dictionary counting during how
                                   many shifts a nurse was assigned to them

        These feel unneeded:
        - least_assigned_nurse_per_patient: a dictionary with for each patient, a tuple of a nurse id and the number of times
                                 this nurse was assigned to the patient. This is the least assigned nurse.

        Newly added (for faster SwapPatient moves):
        - patient_shift_partition: A dictionary with for each patient pair (p1,p2) the partition of all shifts into the
                                   p1_shifts, p2_shifts and overlapping_shifts
        - all_patient_pairs: A list of pairs of patients (p1,p2) that overlap in at least one shift. The index of p1
                             is always lower than that of p2. (same for each solution)
        - valid_patient_pairs: A subset of all_patient_pairs, where pairs are removed where both patients are in the
                               same room or are in rooms incompatible for each other. (depends on current solution)
        - valid_patient_pairs_indices: A dictionary that for each pair in valid_patient_pairs has the corresponding
                                       index in that list. This is for faster lookups.

        For faster RemoveNurse moves:
        - RemoveNurse_options: dictionary with for each patient, a list of [(n,candidate_assignments),...]
        - patients_to_update: set of patients that need updating
    """
    PR_assignment, NR_assignment = solution

    # cap_penalty_weight
    cap_penalty_weight = 100 * max(instance.weights.values())

    # note that patients_per_rooms contains the same patients for the early, late and night shift,
    # but this makes it easier in the computation later
    patients_per_room = {(r, s): [] for r in instance.room_ids for s in instance.all_shifts}
    for p in instance.patient_ids:
        assigned_room = PR_assignment[p]
        for s in instance.patients[p]["shifts"]:
            patients_per_room[assigned_room, s].append(p)

    nurse_per_room = dict()
    for n in instance.nurse_ids:
        for s in instance.nurses[n]["shifts"]:
            for r in NR_assignment[n, s]:
                if (r, s) not in nurse_per_room:
                    nurse_per_room[r, s] = n
                else:
                    raise Exception(
                        f"{(r, s)} was already in nurse_per_room with value {nurse_per_room[r, s]} before assigning {n}")

    gender_numb_per_room = {(r, s, gender): 0 for r in instance.room_ids for s in instance.early_shifts
                            for gender in ["A", "B"]}
    for p in instance.patient_ids:
        assigned_room = PR_assignment[p]
        gender = instance.patients[p]["gender"]
        for s in instance.patients[p]["shifts"]:
            if s in instance.early_shifts:
                gender_numb_per_room[assigned_room, s, gender] += 1

    gender_assignment = dict()
    for r in instance.room_ids:
        for s in instance.early_shifts:
            if gender_numb_per_room[r, s, "A"] == 0 and gender_numb_per_room[r, s, "B"] == 0:
                gender_assignment[r, s] = None
            elif gender_numb_per_room[r, s, "A"] == 0 and gender_numb_per_room[r, s, "B"] > 0:
                gender_assignment[r, s] = "B"
            elif gender_numb_per_room[r, s, "A"] > 0 and gender_numb_per_room[r, s, "B"] == 0:
                gender_assignment[r, s] = "A"
            else:
                gender_assignment[r, s] = "Both"

    NP_assignment = dict()
    for p in instance.patient_ids:
        assigned_room = PR_assignment[p]
        for s in instance.patients[p]["shifts"]:
            nurse_in_room = nurse_per_room[assigned_room, s]
            NP_assignment[p, s] = nurse_in_room

    workload_per_room = {(r, s): 0 for r in instance.room_ids for s in instance.all_shifts}
    for p in instance.patients:
        assigned_room = PR_assignment[p]
        for s in instance.patients[p]["shifts"]:
            workload_per_room[assigned_room, s] += instance.patients[p]["workload"][s]

    workload_per_nurse = {(n, s): 0 for n in instance.nurse_ids for s in instance.nurses[n]["shifts"]}
    for n in instance.nurse_ids:
        for s in instance.nurses[n]["shifts"]:
            for r in NR_assignment[n, s]:
                workload_per_nurse[n, s] += workload_per_room[r, s]

    min_max_rel_load_per_shift = dict()
    for s in instance.all_shifts:
        min_rel_load = float('inf')
        max_rel_load = 0
        for n in instance.nurses_per_shift[s]:
            max_load = instance.nurses[n]["max_load"][s]
            workload = workload_per_nurse[n, s]
            relative_workload = workload / max_load
            if relative_workload < min_rel_load:
                min_rel_load = relative_workload
            if relative_workload > max_rel_load:
                max_rel_load = relative_workload
        min_max_rel_load_per_shift[s] = (min_rel_load, max_rel_load)

    nurse_count_per_patient = {p: dict() for p in instance.patient_ids}
    for p in instance.patient_ids:
        assigned_room = PR_assignment[p]
        for s in instance.patients[p]["shifts"]:
            assigned_nurse = nurse_per_room[assigned_room, s]
            if assigned_nurse not in nurse_count_per_patient[p]:
                nurse_count_per_patient[p][assigned_nurse] = 1
            else:
                nurse_count_per_patient[p][assigned_nurse] += 1

    patient_shift_partition = dict()
    for i in range(len(instance.patient_ids) - 1):
        for j in range(i + 1, len(instance.patient_ids)):
            p1, p2 = instance.patient_ids[i], instance.patient_ids[j]
            if p1 not in instance.occupant_ids and p2 not in instance.occupant_ids:
                p1_shifts = []
                p2_shifts = []
                overlapping_shifts = []
                for s in instance.patients[p1]['shifts']:
                    if s not in instance.patients[p2]['shifts']:
                        p1_shifts.append(s)
                    else:
                        overlapping_shifts.append(s)
                for s in instance.patients[p2]['shifts']:
                    if s not in instance.patients[p1]['shifts']:
                        p2_shifts.append(s)

                patient_shift_partition[p1, p2] = (p1_shifts, p2_shifts, overlapping_shifts)

    non_occupant_patients = instance.non_occupant_ids
    all_patient_pairs = []
    for i in range(len(non_occupant_patients) - 1):
        p1 = non_occupant_patients[i]
        for j in range(i + 1, len(non_occupant_patients)):
            p2 = non_occupant_patients[j]

            (p1_shifts, p2_shifts, overlapping_shifts) = patient_shift_partition[p1, p2]
            if len(overlapping_shifts) != 0:
                all_patient_pairs.append((p1, p2))

    valid_patient_pairs = []
    valid_patient_pairs_indices = dict()
    i = 0
    for (p1, p2) in all_patient_pairs:
        pair = (p1, p2)
        r1 = PR_assignment[p1]
        r2 = PR_assignment[p2]

        # pair is only valid if they are in different rooms and compatible for both
        if r1 == r2:
            continue
        if r1 in instance.patients[p2]["incompatible_rooms"]:
            continue
        if r2 in instance.patients[p1]["incompatible_rooms"]:
            continue

        valid_patient_pairs.append(pair)
        valid_patient_pairs_indices[pair] = i
        i+=1

    RemoveNurse_options = dict()
    patients_to_update = [p for p in instance.patient_ids]

    # Store datastructures in dictionary
    solution_attributes = dict()
    solution_attributes["cap_penalty_weight"] = cap_penalty_weight
    solution_attributes["patients_per_room"] = patients_per_room
    solution_attributes["nurse_per_room"] = nurse_per_room
    solution_attributes["gender_numb_per_room"] = gender_numb_per_room
    solution_attributes["gender_assignment"] = gender_assignment
    solution_attributes["NP_assignment"] = NP_assignment
    solution_attributes["workload_per_room"] = workload_per_room
    solution_attributes["workload_per_nurse"] = workload_per_nurse
    solution_attributes["nurse_count_per_patient"] = nurse_count_per_patient
    solution_attributes["min_max_rel_load_per_shift"] = min_max_rel_load_per_shift
    solution_attributes["patient_shift_partition"] = patient_shift_partition
    solution_attributes["all_patient_pairs"] = all_patient_pairs
    solution_attributes["valid_patient_pairs"] = valid_patient_pairs
    solution_attributes["valid_patient_pairs_indices"] = valid_patient_pairs_indices
    solution_attributes["RemoveNurse_options"] = RemoveNurse_options
    solution_attributes["patients_to_update"] = patients_to_update

    return solution_attributes

# These two functions help with the updating of valid_patient_pairs_indices
def add_pair(pair, patient_pairs, patient_pairs_indices):
    """Add a pair to the patient_pairs list and update the indices"""
    patient_pairs_indices[pair] = len(patient_pairs)
    patient_pairs.append(pair)
    return patient_pairs, patient_pairs_indices

def remove_pair(pair, patient_pairs, patient_pairs_indices):
    """Remove a pair to the patient_pairs list and update the indices"""
    index = patient_pairs_indices.pop(pair) # get the index of the pair
    final_entry = patient_pairs[-1]
    patient_pairs[index] = final_entry # place last element in place of removed pair (prevents reshuffling)
    patient_pairs.pop()
    if final_entry != pair:# update index of last element (if it was not removed)
        patient_pairs_indices[final_entry] = index
    return patient_pairs, patient_pairs_indices

#####################
#  MovePatient move #
#####################

def delta_eval_MovePatient(instance, p, r1, r2, solution_attributes):
    # note that patient p must be assigned to room r1 otherwise this calculation is obviously wrong

    # retrieve values for faster lookup
    patients_per_room = solution_attributes["patients_per_room"]
    nurse_per_room = solution_attributes["nurse_per_room"]
    gender_numb_per_room = solution_attributes["gender_numb_per_room"]
    gender_assignment = solution_attributes["gender_assignment"]
    nurse_count_per_patient = solution_attributes["nurse_count_per_patient"]
    workload_per_nurse = solution_attributes["workload_per_nurse"]
    min_max_rel_load_per_shift = solution_attributes["min_max_rel_load_per_shift"]

    # define the weights
    weights = instance.weights
    continuity_weight = weights["Continuity"]
    gender_weight = weights["Gender-Mixing"]
    skill_weight = weights["Skill Requirements"]
    workload_weight = weights["Workload Violation"]
    imbalance_weight = weights["Workload Imbalance"]
    cap_penalty_weight = solution_attributes["cap_penalty_weight"]

    # dprint(f"\nCalculating delta evaluation for moving patient {p} from {r1} to {r2}...")
    obj_delta = 0
    penalty_delta = 0

    # Capacity penalty
    delta_capacity = 0
    # dprint("\nCapacity delta:")
    C_r1 = instance.room_capacities[r1]
    C_r2 = instance.room_capacities[r2]
    for s in instance.patients[p]["shifts"]:
        if s in instance.early_shifts:
            patients_in_r1 = len(patients_per_room[r1, s])
            patients_in_r2 = len(patients_per_room[r2, s])
            # increase if r1 under cap and r2 over cap
            if patients_in_r1 <= C_r1 and patients_in_r2 >= C_r2:
                delta_capacity += cap_penalty_weight

            # decreases if r1 over cap and r2 under cap
            if patients_in_r1 > C_r1 and patients_in_r2 < C_r2:
                delta_capacity -= cap_penalty_weight
    #             dprint(f"\tdelta_capacity -= {capacity_penalty}")
    # dprint(f"Delta capacity: {delta_capacity}")
    penalty_delta += delta_capacity

    # Continuity of care
    if continuity_weight != 0:
        prev_CoC = len(nurse_count_per_patient[p].keys())
        new_nurses = dict()
        # since patient was moved, must recalculate the nurse_assignment
        for s in instance.patients[p]["shifts"]:
            assigned_nurse = nurse_per_room[r2, s]
            if assigned_nurse not in new_nurses:
                new_nurses[assigned_nurse] = 1
            else:
                new_nurses[assigned_nurse] += 1
        delta_continuity = (len(new_nurses) - prev_CoC) * continuity_weight
        obj_delta += delta_continuity

    # Gender-mixing penalty
    if gender_weight != 0:
        delta_gender = 0
        gender = instance.patients[p]["gender"]
        if gender == "A":
            opp_gender = "B"
        else:
            opp_gender = "A"

        for s in instance.patients[p]["shifts"]:
            if s in instance.early_shifts:
                # increases if r2 is assigned the opposite gender
                if gender_assignment[r2, s] == opp_gender:
                    delta_gender += gender_weight

                # decreases if r1 was both and p was only of their gender
                if gender_assignment[r1, s] == "Both":
                    if gender_numb_per_room[r1, s, gender] == 1:
                        delta_gender -= gender_weight
        obj_delta += delta_gender

    # Skill requirements
    if skill_weight != 0:
        # dprint("\nSkill requirement delta:")
        prev_skill = 0
        new_skill = 0

        for s in instance.patients[p]["shifts"]:
            skill_req = instance.patients[p]["skill"][s]
            assigned_nurse = nurse_per_room[r1, s]
            nurse_skill = instance.nurses[assigned_nurse]["skill"]
            if nurse_skill < skill_req:
                prev_skill += skill_req - nurse_skill

            assigned_nurse = nurse_per_room[r2, s]
            nurse_skill = instance.nurses[assigned_nurse]["skill"]
            if nurse_skill < skill_req:
                new_skill += skill_req - nurse_skill

        # dprint(f"New skill violation = {new_skill}")
        delta_skill = (new_skill - prev_skill) * skill_weight
        # dprint(f"Delta skill: {delta_skill} = {new_skill - prev_skill} * {instance.weights[1]}")
        obj_delta += delta_skill

    # Workload violations and imbalance
    if workload_weight != 0 or imbalance_weight != 0:
        # dprint("\nWorkload violation delta:")

        delta_workload_vio = 0
        delta_workload_imbal = 0
        for s in instance.patients[p]["shifts"]:
            n1 = nurse_per_room[r1, s]
            n2 = nurse_per_room[r2, s]
            # dprint(f"Shift {s}: nurse {n1} in room {r1} and nurse {n2} in room {r2}")
            if n1 != n2:  # if they are the same the workload does not change
                patient_workload = instance.patients[p]["workload"][s]

                w1 = workload_per_nurse[n1, s]
                w2 = workload_per_nurse[n2, s]
                max_load1 = instance.nurses[n1]["max_load"][s]
                max_load2 = instance.nurses[n2]["max_load"][s]
                w1_rel_old = w1 / max_load1
                w2_rel_old = w2 / max_load2

                w1_new = w1 - patient_workload
                w1_rel_new = w1_new / max_load1
                w2_new = w2 + patient_workload
                w2_rel_new = w2_new / max_load2

                ## CALCULATE THE WORKLOAD VIOLATION
                # calculate the old workload violation
                old_workload_vio = 0
                w_diff1 = w1 - max_load1
                if w_diff1 > 0:
                    old_workload_vio += w_diff1
                w_diff2 = w2 - max_load2
                if w_diff2 > 0:
                    old_workload_vio += w_diff2

                # calculate the new workload violation
                new_workload_vio = 0
                w_diff1_new = w1_new - max_load1
                if w_diff1_new > 0:
                    new_workload_vio += w_diff1_new
                w_diff2_new = w2_new - max_load2
                if w_diff2_new > 0:
                    new_workload_vio += w_diff2_new

                delta_workload_vio += (new_workload_vio - old_workload_vio) * workload_weight

                ## CALCULATE THE WORKLOAD IMBALANCE
                cur_min_rel_load, cur_max_rel_load = min_max_rel_load_per_shift[s]
                prev_imbalance = cur_max_rel_load - cur_min_rel_load

                # if increasing nurse had lowest rel workload or decreasing had highest rel workload,
                # the workload imbalance changes and a different nurse could become min or max
                if abs(cur_max_rel_load - w1_rel_old) < 10e-6 or abs(cur_min_rel_load - w2_rel_old) < 10e-6:
                    new_min_rel_load = float('inf')
                    new_max_rel_load = 0
                    for n in instance.nurses_per_shift[s]:
                        if n == n1:
                            relative_workload = w1_rel_new
                        elif n == n2:
                            relative_workload = w2_rel_new
                        else:
                            workload = workload_per_nurse[n, s]
                            max_load = instance.nurses[n]["max_load"][s]
                            relative_workload = workload / max_load
                        if relative_workload < new_min_rel_load:
                            new_min_rel_load = relative_workload
                        if relative_workload > new_max_rel_load:
                            new_max_rel_load = relative_workload
                else:
                    # there are other nurses that have either the same or more extreme relative workloads
                    # This objective only changes if the updated workloads are more extreme
                    if w2_rel_new > cur_max_rel_load:
                        new_max_rel_load = w2_rel_new
                    else:
                        new_max_rel_load = cur_max_rel_load
                    if w1_rel_new < cur_min_rel_load:
                        new_min_rel_load = w1_rel_new
                    else:
                        new_min_rel_load = cur_min_rel_load
                new_imbalance = new_max_rel_load - new_min_rel_load
                delta_workload_imbal += (new_imbalance - prev_imbalance) * imbalance_weight

        obj_delta += delta_workload_vio
        obj_delta += delta_workload_imbal

    # dprint(f"\nTotal delta: {obj_delta}")
    return obj_delta, penalty_delta

def update_solution_attributes_MovePatient(instance, p, r1, r2, solution, solution_attributes):
    """
    Speeds up the updating of the solution attributes by only updating what has changed.
    The following items change:
        - patients_per_room
        - gender_numb_per_room
        - gender_assignment
        - NP_assignment
        - workload_per_room
        - workload_per_nurse
        - min_max_rel_load_per_shift
        - nurse_count_per_patient
    """
    patients_per_room = solution_attributes["patients_per_room"]
    gender_numb_per_room = solution_attributes["gender_numb_per_room"]
    gender_assignment = solution_attributes["gender_assignment"]
    NP_assignment = solution_attributes["NP_assignment"]
    workload_per_room = solution_attributes["workload_per_room"]
    workload_per_nurse = solution_attributes["workload_per_nurse"]
    min_max_rel_load_per_shift = solution_attributes["min_max_rel_load_per_shift"]
    nurse_count_per_patient = solution_attributes["nurse_count_per_patient"]
    valid_patient_pairs = solution_attributes["valid_patient_pairs"]
    valid_patient_pairs_indices = solution_attributes["valid_patient_pairs_indices"]
    patients_to_update = solution_attributes["patients_to_update"]

    # other useful things
    patient_shifts = instance.patients[p]["shifts"]
    gender = instance.patients[p]["gender"]
    patient_workloads = instance.patients[p]["workload"]
    nurse_per_room = solution_attributes["nurse_per_room"]
    nurse_count = dict()

    for s in patient_shifts:
        # update patients_per_room
        patients_per_room[r1,s].remove(p)
        patients_per_room[r2,s].append(p)

        # update gender_numb_per_room
        if s in instance.early_shifts:
            gender_numb_per_room[r1,s,gender] -= 1
            gender_numb_per_room[r2,s,gender] += 1

        # update gender_assignment
        if s in instance.early_shifts:
            if gender_numb_per_room[r1,s,"A"] == 0 and gender_numb_per_room[r1,s,"B"] == 0:
                gender_assignment[r1, s] = None
            elif gender_numb_per_room[r1,s,"A"] == 0 and gender_numb_per_room[r1,s,"B"] > 0:
                gender_assignment[r1, s] = "B"
            elif gender_numb_per_room[r1,s,"A"] > 0 and gender_numb_per_room[r1,s,"B"] == 0:
                gender_assignment[r1, s] = "A"
            else:
                gender_assignment[r1, s] = "Both"

            if gender_numb_per_room[r2,s,"A"] == 0 and gender_numb_per_room[r2,s,"B"] == 0:
                gender_assignment[r2, s] = None
            elif gender_numb_per_room[r2,s,"A"] == 0 and gender_numb_per_room[r2,s,"B"] > 0:
                gender_assignment[r2, s] = "B"
            elif gender_numb_per_room[r2,s,"A"] > 0 and gender_numb_per_room[r2,s,"B"] == 0:
                gender_assignment[r2, s] = "A"
            else:
                gender_assignment[r2, s] = "Both"

        # update NP_assignment
        nurse_in_room = nurse_per_room[r2,s]
        NP_assignment[p,s] = nurse_in_room

        # update workload_per_room
        workload_per_room[r1,s] -= patient_workloads[s]
        workload_per_room[r2,s] += patient_workloads[s]

        # update workload_per_nurse
        n1 = nurse_per_room[r1,s]
        n2 = nurse_per_room[r2,s]
        if n1 != n2:
            workload_per_nurse[n1,s] -= patient_workloads[s]
            workload_per_nurse[n2,s] += patient_workloads[s]

        # update min_max_rel_load_per_shift
        new_min_rel_load = float('inf')
        new_max_rel_load = 0
        for n in instance.nurses_per_shift[s]:
            workload = workload_per_nurse[n, s]
            max_load = instance.nurses[n]["max_load"][s]
            relative_workload = workload / max_load
            if relative_workload < new_min_rel_load:
                new_min_rel_load = relative_workload
            if relative_workload > new_max_rel_load:
                new_max_rel_load = relative_workload
        min_max_rel_load_per_shift[s] = (new_min_rel_load, new_max_rel_load)

        # update nurse_count_per_patient & least_assigned_nurse
        n = nurse_per_room[r2,s]
        if n not in nurse_count:
            nurse_count[n] = 1
        else:
            nurse_count[n] += 1
    nurse_count_per_patient[p] = nurse_count


    # update valid_patient_pairs and valid_patient_pairs_indices
    PR_assignment = solution[0]
    all_patient_pairs = solution_attributes["all_patient_pairs"]
    for pair in all_patient_pairs:
        patient1, patient2 = pair
        if p == patient1 or p == patient2:
            valid_pair = True
            r1, r2 = PR_assignment[patient1], PR_assignment[patient2]
            if r1 == r2:
                valid_pair = False
            if r1 in instance.patients[patient2]["incompatible_rooms"]:
                valid_pair = False
            if r2 in instance.patients[patient1]["incompatible_rooms"]:
                valid_pair = False

            pair_in_list = pair in valid_patient_pairs_indices

            if valid_pair and not pair_in_list:
                valid_patient_pairs, valid_patient_pairs_indices = add_pair(pair, valid_patient_pairs,
                                                                            valid_patient_pairs_indices)
            elif not valid_pair and pair_in_list:
                valid_patient_pairs, valid_patient_pairs_indices = remove_pair(pair, valid_patient_pairs,
                                                                               valid_patient_pairs_indices)
    # update patients_to_update
    if p not in patients_to_update:
        patients_to_update.append(p)

    # store all items again
    solution_attributes["patients_per_room"] = patients_per_room
    solution_attributes["gender_numb_per_room"] = gender_numb_per_room
    solution_attributes["gender_assignment"] = gender_assignment
    solution_attributes["NP_assignment"] = NP_assignment
    solution_attributes["workload_per_room"] = workload_per_room
    solution_attributes["workload_per_nurse"] = workload_per_nurse
    solution_attributes["min_max_rel_load_per_shift"] = min_max_rel_load_per_shift
    solution_attributes["nurse_count_per_patient"] = nurse_count_per_patient
    solution_attributes["valid_patient_pairs"] = valid_patient_pairs
    solution_attributes["valid_patient_pairs_indices"] = valid_patient_pairs_indices
    solution_attributes["patients_to_update"] = patients_to_update
    return solution_attributes

#####################
# SwapPatients move #
#####################

def delta_eval_SwapPatients(instance, p1, p2, r1, r2, solution_attributes):
    """
    Calculates the change in objective value (including penalty for infeasible solutions) when swapping
    patients p1 and p2 between corresponding rooms r1 and r2.
    """
    patients_per_room = solution_attributes["patients_per_room"]
    nurse_per_room = solution_attributes["nurse_per_room"]
    gender_numb_per_room = solution_attributes["gender_numb_per_room"]
    gender_assignment = solution_attributes["gender_assignment"]
    nurse_count_per_patient = solution_attributes["nurse_count_per_patient"]
    workload_per_nurse = solution_attributes["workload_per_nurse"]
    min_max_rel_load_per_shift = solution_attributes["min_max_rel_load_per_shift"]
    patient_shift_partition = solution_attributes["patient_shift_partition"]

    # get patient workload and skill information
    p1_workloads = instance.patients[p1]["workload"]
    p2_workloads = instance.patients[p2]["workload"]
    p1_skills = instance.patients[p1]["skill"]
    p2_skills = instance.patients[p2]["skill"]

    # get all shift information
    p1_shifts = instance.patients[p1]["shifts"]
    p2_shifts = instance.patients[p2]["shifts"]
    p1_only_shifts, p2_only_shifts, overlapping_shifts = patient_shift_partition[p1,p2]

    obj_delta = 0
    penalty_delta = 0

    ##################################
    #            CAPACITY            #
    ##################################

    delta_capacity = 0
    cap_penalty_weight = solution_attributes["cap_penalty_weight"]
    # dprint("\nCapacity delta:")
    # for the overlapping shifts, the capacity does not change
    if len(p1_only_shifts) != 0:
        # dprint(f"Patient {p1}'s shifts:")
        for s in p1_only_shifts:
            if s in instance.early_shifts:
                patients_in_r1 = len(patients_per_room[r1, s])
                patients_in_r2 = len(patients_per_room[r2, s])
                C_r1 = instance.room_capacities[r1]
                C_r2 = instance.room_capacities[r2]
                # dprint(f"Shift {s}: {patients_in_r1} patients in {r1} (cap {C_r1}) and "
                #        f"{patients_in_r2} patients in {r2} (cap {C_r2})")
                if patients_in_r1 <= C_r1 and patients_in_r2 >= C_r2:
                    delta_capacity += cap_penalty_weight
                    # dprint(f"\tdelta_capacity += {capacity_penalty}")
                if patients_in_r1 > C_r1 and patients_in_r2 < C_r2:
                    delta_capacity -= cap_penalty_weight
                    # dprint(f"\tdelta_capacity -= {capacity_penalty}")
    if len(p2_only_shifts) != 0:
        # dprint(f"Patient {p2}'s shifts:")
        for s in p2_only_shifts:
            if s in instance.early_shifts:
                patients_in_r1 = len(patients_per_room[r1, s])
                patients_in_r2 = len(patients_per_room[r2, s])
                C_r1 = instance.room_capacities[r1]
                C_r2 = instance.room_capacities[r2]
                # dprint(f"Shift {s}: {patients_in_r1} patients in {r1} (cap {C_r1}) and "
                #        f"{patients_in_r2} patients in {r2} (cap {C_r2})")
                if patients_in_r2 <= C_r2 and patients_in_r1 >= C_r1:
                    delta_capacity += cap_penalty_weight
                    # dprint(f"\tdelta_capacity += {capacity_penalty}")
                if patients_in_r2 > C_r2 and patients_in_r1 < C_r1:
                    delta_capacity -= cap_penalty_weight
                    # dprint(f"\tdelta_capacity -= {capacity_penalty}")
    # dprint(f"Delta capacity: {delta_capacity}")
    penalty_delta += delta_capacity

    ##################################
    #         GENDER-MIXING          #
    ##################################

    delta_gender = 0
    # dprint("\nGender-mixing delta:")
    g1 = instance.patients[p1]['gender']
    g2 = instance.patients[p2]['gender']
    # dprint(f"Patient {p1} has gender {g1} and "
    #        f"patient {p2} has gender {g2}")
    if g1 != g2 and len(overlapping_shifts) != 0:  # if they are the same, nothing changes in the overlapping shifts
        # dprint("Overlapping shifts:")
        for s in overlapping_shifts:
            if s in instance.early_shifts:
                # dprint(f"Shift {s}: \troom {r1} has {gender_numb_per_room[r1,s,'A']} males and {gender_numb_per_room[r1,s,'B']} females")
                # dprint(f"\t\t\troom {r2} has {gender_numb_per_room[r2,s,'A']} males and {gender_numb_per_room[r2,s,'B']} females")
                # if p1 is the final person of the gender while the room contained both genders, then the room becomes single gender
                if gender_numb_per_room[r1, s, g1] == 1 and gender_assignment[r1, s] == "Both":
                    delta_gender -= instance.weights["Gender-Mixing"]
                    # dprint(f"\tdelta_gender -= {gender_penalty}")
                # if p2 is the final person of the gender while the room contained both genders, then the room becomes single gender
                if gender_numb_per_room[r2, s, g2] == 1 and gender_assignment[r2, s] == "Both":
                    delta_gender -= instance.weights["Gender-Mixing"]
                    # dprint(f"\tdelta_gender -= {gender_penalty}")

                # when does it increase?
                # if the room has only one gender with > 1 patients of that gender
                if gender_assignment[r1, s] != "Both":  # we know it is not None
                    if gender_numb_per_room[r1, s, g1] > 1:
                        delta_gender += instance.weights["Gender-Mixing"]
                        # dprint(f"\tdelta_gender += {gender_penalty}")
                if gender_assignment[r2, s] != "Both":  # we know it is not None
                    if gender_numb_per_room[r2, s, g2] > 1:
                        delta_gender += instance.weights["Gender-Mixing"]
                        # dprint(f"\tdelta_gender += {gender_penalty}")
    if len(p1_only_shifts) != 0:
        # dprint(f"Patient {p1}'s shifts:")
        for s in p1_only_shifts:
            if s in instance.early_shifts:
                # dprint(f"Shift {s}: \troom {r1} has {gender_numb_per_room[r1, s, 'A']} males and {gender_numb_per_room[r1, s, 'B']} females")
                # dprint(f"\t\t\troom {r2} has {gender_numb_per_room[r2, s, 'A']} males and {gender_numb_per_room[r2, s, 'B']} females")
                if gender_assignment[r2, s] != "Both" and gender_assignment[r2, s] is not None:
                    if g1 != gender_assignment[r2, s]:
                        delta_gender += instance.weights["Gender-Mixing"]
                        # dprint(f"\tdelta_gender += {gender_penalty}")
                if gender_assignment[r1, s] == "Both":
                    if gender_numb_per_room[r1, s, g1] == 1:
                        delta_gender -= instance.weights["Gender-Mixing"]
                        # dprint(f"\tdelta_gender -= {gender_penalty}")
    if len(p2_only_shifts) != 0:
        # dprint(f"Patient {p2}'s shifts:")
        for s in p2_only_shifts:
            if s in instance.early_shifts:
                # dprint(f"Shift {s}: \troom {r1} has {gender_numb_per_room[r1, s, 'A']} males and {gender_numb_per_room[r1, s, 'B']} females")
                # dprint(f"\t\t\troom {r2} has {gender_numb_per_room[r2, s, 'A']} males and {gender_numb_per_room[r2, s, 'B']} females")
                if gender_assignment[r1, s] != "Both" and gender_assignment[r1, s] is not None:
                    if g2 != gender_assignment[r1, s]:
                        delta_gender += instance.weights["Gender-Mixing"]
                        # dprint(f"\tdelta_gender += {gender_penalty}")
                if gender_assignment[r2, s] == "Both":
                    if gender_numb_per_room[r2, s, g2] == 1:
                        delta_gender -= instance.weights["Gender-Mixing"]
                        # dprint(f"\tdelta_gender -= {gender_penalty}")
    # dprint(f"Delta gender: {delta_gender}")
    obj_delta += delta_gender

    ##################################
    #       CONTINUITY OF CARE       #
    ##################################

    # Continuity of care
    if instance.weights["Continuity"] != 0:
        # dprint("\nContinuity of care delta:")
        prev_CoC_p1 = len(nurse_count_per_patient[p1].keys())
        # dprint(f"Prev_value p1 = {prev_CoC_p1}")
        new_nurses_p1 = dict()
        for s in p1_shifts:
            assigned_nurse = nurse_per_room[r2, s]
            # dprint(f"Shift {s}: nurse {assigned_nurse} in room {r2}")
            if assigned_nurse not in new_nurses_p1:
                new_nurses_p1[assigned_nurse] = 1
            else:
                new_nurses_p1[assigned_nurse] += 1
        # dprint(f"New number of nurses: {len(new_nurses)}")
        delta_continuity_p1 = (len(new_nurses_p1) - prev_CoC_p1) * instance.weights["Continuity"]
        # dprint(f"Delta continuity {p1}: {delta_continuity_p1} = {len(new_nurses) - prev_CoC_p1} * {instance.weights[0]}\n")
        obj_delta += delta_continuity_p1

        prev_CoC_p2 = len(nurse_count_per_patient[p2].keys())
        # dprint(f"Prev_value p2 = {prev_CoC_p2}")
        new_nurses_p2 = dict()
        for s in p2_shifts:
            assigned_nurse = nurse_per_room[r1, s]
            # dprint(f"Shift {s}: nurse {assigned_nurse} in room {r1}")
            if assigned_nurse not in new_nurses_p2:
                new_nurses_p2[assigned_nurse] = 1
            else:
                new_nurses_p2[assigned_nurse] += 1
        # dprint(f"New number of nurses: {len(new_nurses)}")
        delta_continuity_p2 = (len(new_nurses_p2) - prev_CoC_p2) * instance.weights["Continuity"]
        # dprint(
        # f"Delta continuity {p2}: {delta_continuity_p2} = {len(new_nurses) - prev_CoC_p2} * {instance.weights[0]}")
        obj_delta += delta_continuity_p2


    ##################################
    #       SKILL REQUIREMENT        #
    ##################################

    if instance.weights["Skill Requirements"] != 0:
        # dprint("\nSkill requirement delta:")

        # Skill for p1
        prev_skill_p1 = 0
        new_skill_p1 = 0
        for s in p1_shifts:
            skill_req = p1_skills[s]

            # old skill requirement objective (might be a good idea to also store this using solution_attributes)
            assigned_nurse = nurse_per_room[r1, s]
            nurse_skill = instance.nurses[assigned_nurse]["skill"]
            if nurse_skill < skill_req:
                prev_skill_p1 += skill_req - nurse_skill

            # new skill requirement objective
            assigned_nurse = nurse_per_room[r2, s]
            nurse_skill = instance.nurses[assigned_nurse]["skill"]
            if nurse_skill < skill_req:
                new_skill_p1 += skill_req - nurse_skill
        delta_skill_p1 = (new_skill_p1 - prev_skill_p1) * instance.weights["Skill Requirements"]
        obj_delta += delta_skill_p1

        # skill for p2
        prev_skill_p2 = 0
        new_skill_p2 = 0
        for s in p2_shifts:
            skill_req = p2_skills[s]

            # old skill requirement objective (might be a good idea to also store this using solution_attributes)
            assigned_nurse = nurse_per_room[r2, s]
            nurse_skill = instance.nurses[assigned_nurse]["skill"]
            if nurse_skill < skill_req:
                prev_skill_p2 += skill_req - nurse_skill

            # new skill requirement objective
            assigned_nurse = nurse_per_room[r1, s]
            nurse_skill = instance.nurses[assigned_nurse]["skill"]
            if nurse_skill < skill_req:
                new_skill_p2 += skill_req - nurse_skill
        delta_skill_p2 = (new_skill_p2 - prev_skill_p2) * instance.weights["Skill Requirements"]
        obj_delta += delta_skill_p2
        # dprint(f"Skill requirement delta: {delta_skill_p1 + delta_skill_p2}")

    ##################################
    # WORKLOAD VIOLATION & IMBALANCE #
    ##################################

    if instance.weights["Workload Violation"] != 0 or instance.weights["Workload Imbalance"] != 0:
        # dprint("\nWorkload violation delta:")
        delta_workload_vio = 0
        delta_workload_imbal = 0
        if len(overlapping_shifts) != 0:
            for s in overlapping_shifts:
                n1 = nurse_per_room[r1, s]
                n2 = nurse_per_room[r2, s]
                max_load1 = instance.nurses[n1]["max_load"][s]
                max_load2 = instance.nurses[n2]["max_load"][s]
                # dprint(f"Shift {s}: nurse {n1} in room {r1} and nurse {n2} in room {r2}")
                if n1 != n2:  # if the nurse is the same, nothing changes
                    p1_workload = p1_workloads[s]
                    p2_workload = p2_workloads[s]
                    # dprint(f"\tPatient {p1} has workload {p1_workload} and "
                    #        f"Patient {p2} has workload {p2_workload}")
                    if p1_workload != p2_workload:  # if the workload is the same, nothing changes
                        w1 = workload_per_nurse[n1, s]
                        w2 = workload_per_nurse[n2, s]
                        w1_rel_old = w1 / max_load1
                        w2_rel_old = w2 / max_load2

                        w1_new = w1 - p1_workload + p2_workload
                        w1_rel_new = w1_new / max_load1
                        w2_new = w2 + p1_workload - p2_workload
                        w2_rel_new = w2_new / max_load2

                        ## CALCULATE THE WORKLOAD VIOLATIONS
                        # calculate the old workload violation
                        old_workload_vio = 0
                        w_diff1 = w1 - max_load1
                        if w_diff1 > 0:
                            old_workload_vio += w_diff1
                        w_diff2 = w2 - max_load2
                        if w_diff2 > 0:
                            old_workload_vio += w_diff2

                        # calculate the new workload violation
                        new_workload_vio = 0
                        w_diff1_new = w1_new - max_load1
                        if w_diff1_new > 0:
                            new_workload_vio += w_diff1_new
                        w_diff2_new = w2_new - max_load2
                        if w_diff2_new > 0:
                            new_workload_vio += w_diff2_new
                        delta_workload_vio += (new_workload_vio - old_workload_vio) * instance.weights[
                            "Workload Violation"]

                        ## CALCULATE THE WORKLOAD IMBALANCE
                        cur_min_rel_load, cur_max_rel_load = min_max_rel_load_per_shift[s]
                        prev_imbalance = cur_max_rel_load - cur_min_rel_load

                        # determine which nurses workload increases
                        if w1_new > w1:
                            rel_load_increasing = w1_rel_new
                            rel_load_increasing_old = w1_rel_old
                            rel_load_decreasing = w2_rel_new
                            rel_load_decreasing_old = w2_rel_old
                        else:
                            rel_load_increasing = w2_rel_new
                            rel_load_increasing_old = w2_rel_old
                            rel_load_decreasing = w1_rel_new
                            rel_load_decreasing_old = w1_rel_old

                        # print(f"prev_imbalance = {cur_max_rel_load:.3f} - {cur_min_rel_load:.3f}\t")
                        # print(f"w1_rel_old = {w1_rel_old:.3f}, w1_rel_new = {w1_rel_new:.3f}")
                        # print(f"w2_rel_old = {w2_rel_old:.3f}, w2_rel_new = {w2_rel_new:.3f}")
                        # print(f"rel_load_increasing = {rel_load_increasing:.3f}, rel_load_decreasing = {rel_load_decreasing:.3f}")
                        # print(f"rel_load_increasing_old = {rel_load_increasing_old:.3f}, rel_load_decreasing_old = {rel_load_decreasing_old:.3f}")

                        # if increasing nurse had lowest rel workload or decreasing had highest rel workload,
                        # the workload imbalance changes and a different nurse could become min or max
                        if abs(cur_max_rel_load - rel_load_decreasing_old) < 10e-6 or abs(
                                cur_min_rel_load - rel_load_increasing_old) < 10e-6:
                            # print(f"Route A")
                            new_min_rel_load = float('inf')
                            new_max_rel_load = 0
                            for n in instance.nurses_per_shift[s]:
                                if n == n1:
                                    relative_workload = w1_rel_new
                                elif n == n2:
                                    relative_workload = w2_rel_new
                                else:
                                    workload = workload_per_nurse[n, s]
                                    max_load = instance.nurses[n]["max_load"][s]
                                    relative_workload = workload / max_load
                                if relative_workload < new_min_rel_load:
                                    new_min_rel_load = relative_workload
                                if relative_workload > new_max_rel_load:
                                    new_max_rel_load = relative_workload
                        else:
                            # print(f"Route B")
                            # there are other nurses that have either the same or more extreme relative workloads
                            # This objective only changes if the updated workloads are more extreme
                            if rel_load_increasing > cur_max_rel_load:
                                new_max_rel_load = rel_load_increasing
                            else:
                                new_max_rel_load = cur_max_rel_load
                            if rel_load_decreasing < cur_min_rel_load:
                                new_min_rel_load = rel_load_decreasing
                            else:
                                new_min_rel_load = cur_min_rel_load
                        new_imbalance = new_max_rel_load - new_min_rel_load
                        # print(f"new_imbalance = {new_max_rel_load:.3f} - {new_min_rel_load:.3f}")
                        delta_workload_imbal += (new_imbalance - prev_imbalance) * instance.weights[
                            "Workload Imbalance"]

        if len(p1_only_shifts) != 0:
            # dprint(f"Patient {p1}'s shifts:")
            for s in p1_only_shifts:
                n1 = nurse_per_room[r1, s]
                n2 = nurse_per_room[r2, s]
                max_load1 = instance.nurses[n1]["max_load"][s]
                max_load2 = instance.nurses[n2]["max_load"][s]
                if n1 != n2:  # if the nurse is the same, nothing changes
                    p1_workload = p1_workloads[s]
                    # dprint(f"\tPatient {p1} has workload {p1_workload}")

                    w1 = workload_per_nurse[n1, s]
                    w2 = workload_per_nurse[n2, s]
                    w1_rel_old = w1 / max_load1
                    w2_rel_old = w2 / max_load2

                    ## CALCULATE THE WORKLOAD VIOLATION
                    # calculate the old workload violation
                    # new nurse workloads
                    w1_new = w1 - p1_workload
                    w2_new = w2 + p1_workload
                    w1_rel_new = w1_new / max_load1
                    w2_rel_new = w2_new / max_load2

                    ## calculate workload violation
                    old_workload_vio = 0
                    w_diff1 = w1 - max_load1
                    if w_diff1 > 0:
                        old_workload_vio += w_diff1
                    w_diff2 = w2 - max_load2
                    if w_diff2 > 0:
                        old_workload_vio += w_diff2

                    # calculate the new workload violation
                    new_workload_vio = 0
                    w_diff1_new = w1_new - max_load1
                    if w_diff1_new > 0:
                        new_workload_vio += w_diff1_new
                    w_diff2_new = w2_new - max_load2
                    if w_diff2_new > 0:
                        new_workload_vio += w_diff2_new

                    delta_workload_vio += (new_workload_vio - old_workload_vio) * instance.weights[
                        "Workload Violation"]

                    ## CALCULATE THE WORKLOAD IMBALANCE
                    cur_min_rel_load, cur_max_rel_load = min_max_rel_load_per_shift[s]
                    prev_imbalance = cur_max_rel_load - cur_min_rel_load
                    # print(f"{prev_imbalance} = {cur_max_rel_load} - {cur_min_rel_load}")

                    # if increasing nurse had lowest rel workload or decreasing had highest rel workload,
                    # the workload imbalance changes and a different nurse could become min or max
                    if abs(cur_max_rel_load - w1_rel_old) < 10e-6 or abs(cur_min_rel_load - w2_rel_old) < 10e-6:
                        new_min_rel_load = float('inf')
                        new_max_rel_load = 0
                        for n in instance.nurses_per_shift[s]:
                            if n == n1:
                                relative_workload = w1_rel_new
                            elif n == n2:
                                relative_workload = w2_rel_new
                            else:
                                workload = workload_per_nurse[n, s]
                                max_load = instance.nurses[n]["max_load"][s]
                                relative_workload = workload / max_load
                            if relative_workload < new_min_rel_load:
                                new_min_rel_load = relative_workload
                            if relative_workload > new_max_rel_load:
                                new_max_rel_load = relative_workload
                    else:
                        # there are other nurses that have either the same or more extreme relative workloads
                        # This objective only changes if the updated workloads are more extreme
                        if w2_rel_new > cur_max_rel_load:
                            new_max_rel_load = w2_rel_new
                        else:
                            new_max_rel_load = cur_max_rel_load
                        if w1_rel_new < cur_min_rel_load:
                            new_min_rel_load = w1_rel_new
                        else:
                            new_min_rel_load = cur_min_rel_load
                    new_imbalance = new_max_rel_load - new_min_rel_load
                    delta_workload_imbal += (new_imbalance - prev_imbalance) * instance.weights["Workload Imbalance"]
        if len(p2_only_shifts) != 0:
            # dprint(f"Patient {p2}'s shifts:")
            for s in p2_only_shifts:
                n1 = nurse_per_room[r1, s]
                n2 = nurse_per_room[r2, s]
                max_load1 = instance.nurses[n1]["max_load"][s]
                max_load2 = instance.nurses[n2]["max_load"][s]
                if n1 != n2:  # if the nurse is the same, nothing changes
                    p2_workload = p2_workloads[s]

                    w1 = workload_per_nurse[n1, s]
                    w2 = workload_per_nurse[n2, s]
                    w1_rel_old = w1 / max_load1
                    w2_rel_old = w2 / max_load2

                    ## CALCULATE THE WORKLOAD VIOLATION
                    # new nurse workloads
                    w1_new = w1 + p2_workload
                    w2_new = w2 - p2_workload
                    w1_rel_new = w1_new / max_load1
                    w2_rel_new = w2_new / max_load2

                    ## calculate workload violation
                    old_workload_vio = 0
                    w_diff1 = w1 - max_load1
                    if w_diff1 > 0:
                        old_workload_vio += w_diff1
                    w_diff2 = w2 - max_load2
                    if w_diff2 > 0:
                        old_workload_vio += w_diff2

                    # calculate the new workload violation
                    new_workload_vio = 0
                    w_diff1_new = w1_new - max_load1
                    if w_diff1_new > 0:
                        new_workload_vio += w_diff1_new
                    w_diff2_new = w2_new - max_load2
                    if w_diff2_new > 0:
                        new_workload_vio += w_diff2_new

                    delta_workload_vio += (new_workload_vio - old_workload_vio) * instance.weights[
                        "Workload Violation"]

                    ## CALCULATE THE WORKLOAD IMBALANCE
                    cur_min_rel_load, cur_max_rel_load = min_max_rel_load_per_shift[s]
                    prev_imbalance = cur_max_rel_load - cur_min_rel_load
                    # print(f"{prev_imbalance} = {cur_max_rel_load} - {cur_min_rel_load}")

                    # if increasing nurse had lowest rel workload or decreasing had highest rel workload,
                    # the workload imbalance changes and a different nurse could become min or max
                    if abs(cur_max_rel_load - w2_rel_old) < 10e-6 or abs(cur_min_rel_load - w1_rel_old) < 10e-6:
                        new_min_rel_load = float('inf')
                        new_max_rel_load = 0
                        for n in instance.nurses_per_shift[s]:
                            if n == n1:
                                relative_workload = w1_rel_new
                            elif n == n2:
                                relative_workload = w2_rel_new
                            else:
                                workload = workload_per_nurse[n, s]
                                max_load = instance.nurses[n]["max_load"][s]
                                relative_workload = workload / max_load
                            if relative_workload < new_min_rel_load:
                                new_min_rel_load = relative_workload
                            if relative_workload > new_max_rel_load:
                                new_max_rel_load = relative_workload
                    else:
                        # there are other nurses that have either the same or more extreme relative workloads
                        # This objective only changes if the updated workloads are more extreme
                        if w1_rel_new > cur_max_rel_load:
                            new_max_rel_load = w1_rel_new
                        else:
                            new_max_rel_load = cur_max_rel_load
                        if w2_rel_new < cur_min_rel_load:
                            new_min_rel_load = w2_rel_new
                        else:
                            new_min_rel_load = cur_min_rel_load
                    new_imbalance = new_max_rel_load - new_min_rel_load
                    delta_workload_imbal += (new_imbalance - prev_imbalance) * instance.weights["Workload Imbalance"]

        obj_delta += delta_workload_vio
        obj_delta += delta_workload_imbal

    return obj_delta, penalty_delta

def update_solution_attributes_SwapPatients(instance, p1, p2, r1, r2, solution, solution_attributes):
    """
    Speeds up the updating of the solution attributes by only updating what has changed.
    The following items change:
        - patients_per_room
        - gender_numb_per_room
        - gender_assignment
        - NP_assignment
        - workload_per_room
        - workload_per_nurse
        - nurse_count_per_patient
        - valid_patient_pairs
        - valid_patient_pairs_indices
    """
    patients_per_room = solution_attributes["patients_per_room"]
    gender_numb_per_room = solution_attributes["gender_numb_per_room"]
    gender_assignment = solution_attributes["gender_assignment"]
    NP_assignment = solution_attributes["NP_assignment"]
    workload_per_room = solution_attributes["workload_per_room"]
    workload_per_nurse = solution_attributes["workload_per_nurse"]
    min_max_rel_load_per_shift = solution_attributes["min_max_rel_load_per_shift"]
    nurse_count_per_patient = solution_attributes["nurse_count_per_patient"]
    valid_patient_pairs = solution_attributes["valid_patient_pairs"]
    valid_patient_pairs_indices = solution_attributes["valid_patient_pairs_indices"]
    patients_to_update = solution_attributes["patients_to_update"]

    # shifts
    p1_shifts = instance.patients[p1]["shifts"]
    p2_shifts = instance.patients[p2]["shifts"]
    shift_union = set(p1_shifts) | set(p2_shifts)

    # other useful things
    p1_gender = instance.patients[p1]["gender"]
    p2_gender = instance.patients[p2]["gender"]
    p1_workloads = instance.patients[p1]["workload"]
    p2_workloads = instance.patients[p2]["workload"]
    nurse_per_room = solution_attributes["nurse_per_room"]

    # update patients_per_room
    for s in p1_shifts:
        patients_per_room[r1,s].remove(p1)
        patients_per_room[r2,s].append(p1)
    for s in p2_shifts:
        patients_per_room[r2,s].remove(p2)
        patients_per_room[r1,s].append(p2)

    # update gender_numb_per_room
    for s in p1_shifts:
        if s in instance.early_shifts:
            gender_numb_per_room[r1,s,p1_gender] -= 1
            gender_numb_per_room[r2,s,p1_gender] += 1
    for s in p2_shifts:
        if s in instance.early_shifts:
            gender_numb_per_room[r2,s,p2_gender] -= 1
            gender_numb_per_room[r1,s,p2_gender] += 1

    # update gender_assignment
    for s in shift_union:
        if s in instance.early_shifts:
            if gender_numb_per_room[r1,s,"A"] == 0 and gender_numb_per_room[r1,s,"B"] == 0:
                gender_assignment[r1, s] = None
            elif gender_numb_per_room[r1,s,"A"] == 0 and gender_numb_per_room[r1,s,"B"] > 0:
                gender_assignment[r1, s] = "B"
            elif gender_numb_per_room[r1,s,"A"] > 0 and gender_numb_per_room[r1,s,"B"] == 0:
                gender_assignment[r1, s] = "A"
            else:
                gender_assignment[r1, s] = "Both"

            if gender_numb_per_room[r2,s,"A"] == 0 and gender_numb_per_room[r2,s,"B"] == 0:
                gender_assignment[r2, s] = None
            elif gender_numb_per_room[r2,s,"A"] == 0 and gender_numb_per_room[r2,s,"B"] > 0:
                gender_assignment[r2, s] = "B"
            elif gender_numb_per_room[r2,s,"A"] > 0 and gender_numb_per_room[r2,s,"B"] == 0:
                gender_assignment[r2, s] = "A"
            else:
                gender_assignment[r2, s] = "Both"

    # update NP_assignment
    for s in p1_shifts:
        r2_nurse = nurse_per_room[r2,s]
        NP_assignment[p1,s] = r2_nurse
    for s in p2_shifts:
        r1_nurse = nurse_per_room[r1,s]
        NP_assignment[p2,s] = r1_nurse

    # update workload_per_room
    for s in p1_shifts:
        workload_per_room[r1,s] -= p1_workloads[s]
        workload_per_room[r2,s] += p1_workloads[s]
    for s in p2_shifts:
        workload_per_room[r2,s] -= p2_workloads[s]
        workload_per_room[r1,s] += p2_workloads[s]

    # update workload_per_nurse
    for s in p1_shifts:
        n1 = nurse_per_room[r1,s]
        n2 = nurse_per_room[r2,s]
        if n1 != n2:
            workload_per_nurse[n1,s] -= p1_workloads[s]
            workload_per_nurse[n2,s] += p1_workloads[s]
    for s in p2_shifts:
        n1 = nurse_per_room[r1,s]
        n2 = nurse_per_room[r2,s]
        if n1 != n2:
            workload_per_nurse[n1,s] += p2_workloads[s]
            workload_per_nurse[n2,s] -= p2_workloads[s]

    # update min_max_rel_load_per_shift
    for s in shift_union:
        new_min_rel_load = float('inf')
        new_max_rel_load = 0
        for n in instance.nurses_per_shift[s]:
            workload = workload_per_nurse[n, s]
            max_load = instance.nurses[n]["max_load"][s]
            relative_workload = workload / max_load
            if relative_workload < new_min_rel_load:
                new_min_rel_load = relative_workload
            if relative_workload > new_max_rel_load:
                new_max_rel_load = relative_workload
        min_max_rel_load_per_shift[s] = (new_min_rel_load, new_max_rel_load)

    # update nurse_count_per_patient
    nurse_count_p1 = dict()
    nurse_count_p2 = dict()
    for s in p1_shifts:
        n = nurse_per_room[r2,s]
        if n not in nurse_count_p1:
            nurse_count_p1[n] = 1
        else:
            nurse_count_p1[n] += 1
    for s in p2_shifts:
        n = nurse_per_room[r1,s]
        if n not in nurse_count_p2:
            nurse_count_p2[n] = 1
        else:
            nurse_count_p2[n] += 1
    nurse_count_per_patient[p1] = nurse_count_p1
    nurse_count_per_patient[p2] = nurse_count_p2

    # update valid_patient_pairs and valid_patient_pairs_indices
    PR_assignment = solution[0]
    all_patient_pairs = solution_attributes["all_patient_pairs"]
    for pair in all_patient_pairs:
        patient1, patient2 = pair
        if p1 == patient1 or p1 == patient2 or p2 == patient1 or p2 == patient2:
            valid_pair = True
            r1, r2 = PR_assignment[patient1], PR_assignment[patient2]
            if r1 == r2:
                valid_pair = False
            if r1 in instance.patients[patient2]["incompatible_rooms"]:
                valid_pair = False
            if r2 in instance.patients[patient1]["incompatible_rooms"]:
                valid_pair = False

            pair_in_list = pair in valid_patient_pairs_indices

            if valid_pair and not pair_in_list:
                valid_patient_pairs, valid_patient_pairs_indices = add_pair(pair, valid_patient_pairs, valid_patient_pairs_indices)
            elif not valid_pair and pair_in_list:
                valid_patient_pairs, valid_patient_pairs_indices = remove_pair(pair, valid_patient_pairs, valid_patient_pairs_indices)

    # update patients_to_update
    if p1 not in patients_to_update:
        patients_to_update.append(p1)
    if p2 not in patients_to_update:
        patients_to_update.append(p2)

    # store all items again
    solution_attributes["patients_per_room"] = patients_per_room
    solution_attributes["gender_numb_per_room"] = gender_numb_per_room
    solution_attributes["gender_assignment"] = gender_assignment
    solution_attributes["NP_assignment"] = NP_assignment
    solution_attributes["workload_per_room"] = workload_per_room
    solution_attributes["workload_per_nurse"] = workload_per_nurse
    solution_attributes["min_max_rel_load_per_shift"] = min_max_rel_load_per_shift
    solution_attributes["nurse_count_per_patient"] = nurse_count_per_patient
    solution_attributes["valid_patient_pairs"] = valid_patient_pairs
    solution_attributes["valid_patient_pairs_indices"] = valid_patient_pairs_indices
    solution_attributes["patients_to_update"] = patients_to_update
    return solution_attributes

#####################
#  ChangeNurse move #
#####################
def delta_eval_ChangeNurse(instance, r, s, n1, n2, solution_attributes):
    patients_per_room = solution_attributes["patients_per_room"]
    nurse_count_per_patient = solution_attributes["nurse_count_per_patient"]
    workload_per_room = solution_attributes["workload_per_room"]
    workload_per_nurse = solution_attributes["workload_per_nurse"]
    min_max_rel_load_per_shift = solution_attributes["min_max_rel_load_per_shift"]

    # dprint(f"\nCalculating delta change for changing the nurse in room {r} during shift {s} from {n1} to {n2}...")
    obj_delta = 0

    # prevent needless calculations when room is empty
    if len(patients_per_room[r,s]) == 0:
        return 0

    if instance.weights["Continuity"] != 0:
        # dprint("\nContinuity of care delta:")
        delta_continuity = 0
        for p in patients_per_room[r, s]:
            nurse_count = nurse_count_per_patient[p]
            # dprint(f"Patient {p} -> {nurse_count}")
            if nurse_count[n1] == 1:
                # dprint(f"\tNurse {n1} only assigned once -> delta_continuity -= {instance.weights[0]}")
                delta_continuity -= instance.weights["Continuity"]
            if n2 not in nurse_count:
                # dprint(f"\tNurse {n2} not used before -> delta_continuity += {instance.weights[0]}")
                delta_continuity += instance.weights["Continuity"]
        # dprint(f"Delta continuity: {delta_continuity}")
        obj_delta += delta_continuity

    if instance.weights["Skill Requirements"] != 0:
        # dprint("\nSkill requirement delta:")
        n1_skill = instance.nurses[n1]["skill"]
        n2_skill = instance.nurses[n2]["skill"]
        # dprint(f"Nurse {n1} has skill {n1_skill} and nurse {n2} has skill {n2_skill}")
        if n1_skill != n2_skill: # otherwise the objective does not change
            prev_skill = 0
            new_skill = 0
            for p in patients_per_room[r,s]:
                patient_skill = instance.patients[p]['skill'][s]
                # dprint(f"Patient {p}: skill requirement {patient_skill}")
                prev_skill += max(0, patient_skill - n1_skill)
                new_skill += max(0, patient_skill - n2_skill)
            # dprint(f"Delta skill: {(new_skill - prev_skill)} * {instance.weights[1]}")
            obj_delta += (new_skill - prev_skill) * instance.weights["Skill Requirements"]

    if instance.weights["Workload Violation"] != 0 or instance.weights["Workload Imbalance"] != 0:
        # Retrieve nurse information
        w1 = workload_per_nurse[n1, s]
        w2 = workload_per_nurse[n2, s]
        max_load1 = instance.nurses[n1]["max_load"][s]
        max_load2 = instance.nurses[n2]["max_load"][s]
        room_workload = workload_per_room[r, s]

        # compute new workloads
        w1_new = w1 - room_workload
        w1_rel_new = w1_new / max_load1
        w2_new = w2 + room_workload
        w2_rel_new = w2_new / max_load2

        ## Calculate workload violation delta
        prev_workload_vio = max(0, w1 - max_load1) + max(0, w2 - max_load2)
        new_workload_vio = max(0, w1_new - max_load1) + max(0, w2_new - max_load2)
        delta_workload_vio = (new_workload_vio - prev_workload_vio) * instance.weights["Workload Violation"]
        obj_delta += delta_workload_vio


        ## Calculate workload imbalance delta
        # get previous imbalance
        cur_min_rel_load, cur_max_rel_load = min_max_rel_load_per_shift[s]
        prev_imbalance = cur_max_rel_load - cur_min_rel_load

        w1_rel_old = w1 / max_load1
        w2_rel_old = w2 / max_load2
        if abs(cur_max_rel_load - w1_rel_old) < 10e-6 or abs(cur_min_rel_load - w2_rel_old) < 10e-6:
            new_min_rel_load = float('inf')
            new_max_rel_load = 0
            for n in instance.nurses_per_shift[s]:
                if n == n1:
                    relative_workload = w1_rel_new
                elif n == n2:
                    relative_workload = w2_rel_new
                else:
                    workload = workload_per_nurse[n, s]
                    max_load = instance.nurses[n]["max_load"][s]
                    relative_workload = workload / max_load
                if relative_workload < new_min_rel_load:
                    new_min_rel_load = relative_workload
                if relative_workload > new_max_rel_load:
                    new_max_rel_load = relative_workload
        else:
            # there are other nurses that have either the same or more extreme relative workloads
            # This objective only changes if the updated workloads are more extreme
            if w2_rel_new > cur_max_rel_load:
                new_max_rel_load = w2_rel_new
            else:
                new_max_rel_load = cur_max_rel_load
            if w1_rel_new < cur_min_rel_load:
                new_min_rel_load = w1_rel_new
            else:
                new_min_rel_load = cur_min_rel_load
        new_imbalance = new_max_rel_load - new_min_rel_load
        delta_workload_imbal = (new_imbalance - prev_imbalance) * instance.weights["Workload Imbalance"]
        obj_delta += delta_workload_imbal

    return obj_delta

def update_solution_attributes_ChangeNurse(instance, r, s, n1, n2, solution_attributes):
    """
    Speeds up the updating of the solution attributes by only updating what has changed.
    The following items change:
        - nurse_per_room
        - NP_assignment
        - workload_per_nurse
        - min_max_rel_load_per_shift
        - nurse_count_per_patient
    """

    # Things that change
    nurse_per_room = solution_attributes["nurse_per_room"]
    NP_assignment = solution_attributes["NP_assignment"]
    workload_per_nurse = solution_attributes["workload_per_nurse"]
    min_max_rel_load_per_shift = solution_attributes["min_max_rel_load_per_shift"]
    nurse_count_per_patient = solution_attributes["nurse_count_per_patient"]
    patients_to_update = solution_attributes["patients_to_update"]

    # useful for calculations
    patients_per_room = solution_attributes["patients_per_room"]
    workload_per_room = solution_attributes["workload_per_room"]

    # update nurse_per_room
    nurse_per_room[r,s] = n2
    solution_attributes["nurse_per_room"] = nurse_per_room

    # Update NP-assignment
    for p in patients_per_room[r,s]:
        NP_assignment[p,s] = n2
    solution_attributes["NP_assignment"] = NP_assignment

    # Update workload_per_nurse
    room_workload = workload_per_room[r,s]
    workload_per_nurse[n1,s] -= room_workload
    workload_per_nurse[n2,s] += room_workload
    solution_attributes["workload_per_nurse"] = workload_per_nurse

    # Update min_max_rel_load_per_shift
    new_min_rel_load = float('inf')
    new_max_rel_load = 0
    for n in instance.nurses_per_shift[s]:
        workload = workload_per_nurse[n, s]
        max_load = instance.nurses[n]["max_load"][s]
        relative_workload = workload / max_load
        if relative_workload < new_min_rel_load:
            new_min_rel_load = relative_workload
        if relative_workload > new_max_rel_load:
            new_max_rel_load = relative_workload
    min_max_rel_load_per_shift[s] = (new_min_rel_load, new_max_rel_load)
    solution_attributes["min_max_rel_load_per_shift"] = min_max_rel_load_per_shift

    # Update nurse_count_per_patient
    for p in patients_per_room[r,s]:
        # update the counts
        if nurse_count_per_patient[p][n1] == 1:
            nurse_count_per_patient[p].pop(n1)
        else:
            nurse_count_per_patient[p][n1] -= 1

        if n2 not in nurse_count_per_patient[p]:
            nurse_count_per_patient[p][n2] = 1
        else:
            nurse_count_per_patient[p][n2] += 1

    solution_attributes["nurse_count_per_patient"] = nurse_count_per_patient

    # update patients_to_update
    for p in patients_per_room[r, s]:
        if p not in patients_to_update:
            patients_to_update.append(p)
    solution_attributes["patients_to_update"] = patients_to_update

    return solution_attributes

#####################
#  RemoveNurse move #
#####################
def delta_eval_RemoveNurse(instance, p0, r, n1, new_assignments, PR_assignment, solution_attributes):
    patients_per_room = solution_attributes["patients_per_room"]
    nurse_count_per_patient = solution_attributes["nurse_count_per_patient"]
    workload_per_room = solution_attributes["workload_per_room"]
    workload_per_nurse = solution_attributes["workload_per_nurse"]
    min_max_rel_load_per_shift = solution_attributes["min_max_rel_load_per_shift"]

    obj_delta= 0

    # Change in continuity of care
    continuity_delta = -1 # For patient p, decreases by 1
    if instance.weights["Continuity"] != 0:
        for p in instance.patient_ids:
            if p != p0:
                if PR_assignment[p] == r: # Patient in the same room
                    # check the number of shifts that this nurse was replaced
                    number_shifts_replaced = 0
                    new_nurses = []
                    for (s, n2) in new_assignments:
                        if s in instance.patients[p]["shifts"]:
                            number_shifts_replaced += 1
                            if n2 not in new_nurses:
                                new_nurses.append(n2)

                    if number_shifts_replaced != 0:
                        # When can the CoC of a patient go down? Either one of these is true:
                        # - every time that nurse n1 was assigned to patient p0, they are replaced
                        # - there are shifts (non-overlapping with p) that still have nurse n1
                        # only in the first case does the CoC go down
                        nurse_count = nurse_count_per_patient[p]
                        n1_count = nurse_count[n1]
                        if n1_count == number_shifts_replaced:
                            continuity_delta -= 1

                        # When can the CoC go up? If new nurse was not in nurse_count
                        for n2 in new_nurses:
                            if n2 not in nurse_count:
                                continuity_delta += 1
        obj_delta += continuity_delta * instance.weights["Continuity"]

    # Change in skill requirements
    skill_delta = 0
    if instance.weights["Skill Requirements"] != 0:
        n1_skill = instance.nurses[n1]["skill"]
        for (s, n2) in new_assignments:
            n2_skill = instance.nurses[n2]["skill"]
            if n1_skill != n2_skill:
                prev_skill = 0
                new_skill = 0
                for p in patients_per_room[r,s]:
                    patient_skill = instance.patients[p]["skill"][s]
                    prev_skill += max(0, patient_skill - n1_skill)
                    new_skill += max(0, patient_skill - n2_skill)
                skill_delta += (new_skill - prev_skill) * instance.weights["Skill Requirements"]
        obj_delta += skill_delta

    # Change in workload violation and workload imbalance
    workload_delta = 0
    imbalance_delta = 0
    if instance.weights["Workload Violation"] != 0 or instance.weights["Workload Imbalance"] != 0:
        for (s, n2) in new_assignments:
            # retrieve relevant information
            w1 = workload_per_nurse[n1, s]
            w2 = workload_per_nurse[n2, s]
            max_load1 = instance.nurses[n1]["max_load"][s]
            max_load2 = instance.nurses[n2]["max_load"][s]
            room_workload = workload_per_room[r, s]

            # calculate new workloads
            w1_new = w1 - room_workload
            w2_new = w2 + room_workload
            w1_rel_new = w1_new / max_load1
            w2_rel_new = w2_new / max_load2

            ## calculate workload violation delta
            prev_workload_vio = max(0, w1 - max_load1) + max(0, w2 - max_load2)
            new_workload_vio = max(0, w1_new - max_load1) + max(0, w2_new - max_load2)
            workload_delta += (new_workload_vio - prev_workload_vio) * instance.weights["Workload Violation"]

            ## calculate workload imbalance delta
            # get previous imbalance
            cur_min_rel_load, cur_max_rel_load = min_max_rel_load_per_shift[s]
            prev_imbalance = cur_max_rel_load - cur_min_rel_load

            w1_rel_old = w1 / max_load1
            w2_rel_old = w2 / max_load2
            if abs(cur_max_rel_load - w1_rel_old) < 10e-6 or abs(cur_min_rel_load - w2_rel_old) < 10e-6:
                new_min_rel_load = float('inf')
                new_max_rel_load = 0
                for n in instance.nurses_per_shift[s]:
                    if n == n1:
                        relative_workload = w1_rel_new
                    elif n == n2:
                        relative_workload = w2_rel_new
                    else:
                        workload = workload_per_nurse[n, s]
                        max_load = instance.nurses[n]["max_load"][s]
                        relative_workload = workload / max_load
                    if relative_workload < new_min_rel_load:
                        new_min_rel_load = relative_workload
                    if relative_workload > new_max_rel_load:
                        new_max_rel_load = relative_workload
            else:
                # there are other nurses that have either the same or more extreme relative workloads
                # This objective only changes if the updated workloads are more extreme
                if w2_rel_new > cur_max_rel_load:
                    new_max_rel_load = w2_rel_new
                else:
                    new_max_rel_load = cur_max_rel_load
                if w1_rel_new < cur_min_rel_load:
                    new_min_rel_load = w1_rel_new
                else:
                    new_min_rel_load = cur_min_rel_load

            new_imbalance = new_max_rel_load - new_min_rel_load
            imbalance_delta += (new_imbalance - prev_imbalance) * instance.weights["Workload Imbalance"]
        obj_delta += workload_delta
        obj_delta += imbalance_delta

    # print(f"Continuity delta = {continuity_delta}")
    # print(f"Skill requirement delta = {skill_delta}")
    # print(f"Workload Violation delta = {workload_delta}")
    # print(f"Workload Imbalance delta = {imbalance_delta:.3f}")

    return obj_delta

def update_solution_attributes_RemoveNurse(instance, p0, r, n1, new_assignments, solution_attributes):
    """
    Speeds up the updating of the solution attributes by only updating what has changed.
    The following items change:
        - nurse_per_room
        - NP_assignment
        - workload_per_nurse
        - min_max_rel_load_per_shift
        - nurse_count_per_patient
    """
    # Things that change
    nurse_per_room = solution_attributes["nurse_per_room"]
    NP_assignment = solution_attributes["NP_assignment"]
    workload_per_nurse = solution_attributes["workload_per_nurse"]
    min_max_rel_load_per_shift = solution_attributes["min_max_rel_load_per_shift"]
    nurse_count_per_patient = solution_attributes["nurse_count_per_patient"]

    patients_to_update = solution_attributes["patients_to_update"]

    # useful for calculations
    patients_per_room = solution_attributes["patients_per_room"]
    workload_per_room = solution_attributes["workload_per_room"]

    # update nurse_per_room
    for (s, n2) in new_assignments:
        nurse_per_room[r,s] = n2

    # update NP_assignment
    for (s, n2) in new_assignments:
        for p in patients_per_room[r,s]:
            NP_assignment[p,s] = n2

    # update workload_per_nurse
    for (s, n2) in new_assignments:
        r_load = workload_per_room[r,s]
        workload_per_nurse[n1,s] -= r_load
        workload_per_nurse[n2,s] += r_load

    # update min_max_rel_load_per_shift
    for (s, n2) in new_assignments:
        new_min_rel_load = float('inf')
        new_max_rel_load = 0
        for n in instance.nurses_per_shift[s]:
            workload = workload_per_nurse[n, s]
            max_load = instance.nurses[n]["max_load"][s]
            relative_workload = workload / max_load
            if relative_workload < new_min_rel_load:
                new_min_rel_load = relative_workload
            if relative_workload > new_max_rel_load:
                new_max_rel_load = relative_workload
        min_max_rel_load_per_shift[s] = (new_min_rel_load, new_max_rel_load)

    # update nurse_count_per_patient
    updated_patients = set()
    for (s, n2) in new_assignments:
        for p in patients_per_room[r,s]:
            updated_patients.add(p)
            if nurse_count_per_patient[p][n1] == 1:
                nurse_count_per_patient[p].pop(n1)
            else:
                nurse_count_per_patient[p][n1] -= 1

            if n2 not in nurse_count_per_patient[p]:
                nurse_count_per_patient[p][n2] = 1
            else:
                nurse_count_per_patient[p][n2] += 1

    # update patients_to_update
    for (s, n2) in new_assignments:
        for p in patients_per_room[r,s]:
            if p not in patients_to_update:
                patients_to_update.append(p)

    solution_attributes["nurse_per_room"] = nurse_per_room
    solution_attributes["NP_assignment"] = NP_assignment
    solution_attributes["workload_per_nurse"] = workload_per_nurse
    solution_attributes["min_max_rel_load_per_shift"] = min_max_rel_load_per_shift
    solution_attributes["nurse_count_per_patient"] = nurse_count_per_patient

    solution_attributes["patients_to_update"] = patients_to_update
    return solution_attributes
