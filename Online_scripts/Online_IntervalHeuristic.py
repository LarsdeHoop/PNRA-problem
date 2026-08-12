from HelperFunctions import *
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB

class OnlineIntervalHeuristic:
    def __init__(self, instance):
        self.instance = instance
        self.numberOfDays = instance.numberOfDays
        self.L_max_per_size = {'small': 7, 'medium': 2, 'large': 1}
        self.L_max_per_update_shift = dict()
        self.patient_overlap_per_update_shift = dict()

        self.K_per_update_shift = dict()
        self.partition_per_update_shift = dict()
        self.partition_overlap_per_update_shift = dict()

        self.time_budget_per_s_update = dict()
        self.time_limits_per_s_update = dict()
        self.time_limit_per_interval = []
        self.time_spent_per_interval = dict()
        self.numb_timeouts = 0 # counts the number of times the ilp was not solved within the time limit
        self.print_ilp_log = False
        self.print_extra_info = False
        self.print_progress = False

        self.use_proportional_runtime = True

        self.last_interval_considered = 0
        self.status = None
        # we have three possible status codes:
        #   - "done": the heuristic has finished successfully
        #   - "infeasible": the heuristic terminated early because a previous decision led to infeasible ILP
        #   - "timeout": the ILP was not able to find a solution in one interval due to time limit

        self.get_best_partition()

    def execute(self):
        # determine the time budget per s_update
        numb_remaining_days_per_s_update = {s_update: self.instance.numberOfDays - s_update // 3 for s_update in
                                            self.instance.schedule_update_shifts}
        self.time_budget_per_s_update = {s_update: 20 * numb_remaining_days_per_s_update[s_update] for s_update in
                                         self.instance.schedule_update_shifts}

        # initialize things
        fixed_PR_assignment = dict()
        fixed_NR_assignment = dict()
        warm_start_values = dict()

        # retrieve as it is often used
        instance = self.instance

        solution = None
        solution_history = dict()
        self.time_spent_per_interval = {s_update: [] for s_update in instance.schedule_update_shifts}

        if self.print_progress:
            print(f"\tFinished update shifts:", end=" ")
        for i, s_update in enumerate(instance.schedule_update_shifts):
            time_budget = self.time_budget_per_s_update[s_update]
            partition = self.partition_per_update_shift[s_update]
            K = self.K_per_update_shift[s_update]

            if self.use_proportional_runtime:
                total_numb_days = numb_remaining_days_per_s_update[s_update]
                self.time_limit_per_interval = [time_budget * partition[k] / total_numb_days for k in range(K)]
            else:
                self.time_limit_per_interval = [time_budget / K for k in range(K)]

            if self.print_extra_info:
                print(f"Solving from s_update = {s_update} (L_max = {self.L_max_per_update_shift[s_update]} & "
                      f"{len(self.instance.schedulable_patients[s_update])} patients)")
                print(f"\t-> partition = {partition}")

            # determine patients_that are the current occupants
            occupants = instance.occupants_per_update_shift[s_update]
            prev_room_per_occupant = dict()
            if s_update == 0:
                for p in occupants:
                    prev_room_per_occupant[p] = instance.patients[p]["prev_room"]
            else:
                for p in occupants:
                    prev_room_per_occupant[p] = fixed_PR_assignment[p,s_update - 3]

            # initialize nurses_per_patient (for CoC computation)
            temp_PR_assignment = fixed_PR_assignment.copy()
            temp_NR_assignment = {(n,s):[] for s in instance.all_shifts for n in instance.nurses_per_shift[s] if s >= s_update}
            for (n, s) in fixed_NR_assignment:
                temp_NR_assignment[n, s] = fixed_NR_assignment[n, s]

            # set correct values based on fixed values
            nurses_per_patient = {p: [] for p in instance.schedulable_patients[s_update]}
            for p in occupants:
                for s in instance.patients[p]["shifts"]:
                    if s < s_update:  # go over things that are fixed
                        if s in instance.early_shifts:
                            r = fixed_PR_assignment[p, s]
                        elif s in instance.late_shifts:
                            r = fixed_PR_assignment[p, s - 1]
                        else:
                            r = fixed_PR_assignment[p, s - 2]

                        # find which nurse is assigned to room r
                        for n in instance.nurses_per_shift[s]:
                            if r in fixed_NR_assignment[n, s]:
                                if n not in nurses_per_patient[p]:
                                    nurses_per_patient[p].append(n)

            # set_debug(True)
            first_day_of_interval = s_update // 3
            for k in range(K):
                interval_length = partition[k]

                first_shift_of_interval = 3 * first_day_of_interval
                days_in_interval = list(range(first_day_of_interval, first_day_of_interval + interval_length))
                early_shifts_in_interval = [3 * day for day in days_in_interval]
                shifts_in_interval = list(range(3 * first_day_of_interval, 3 * (first_day_of_interval + interval_length)))
                dprint(f"\nConsidering the following days: {days_in_interval}")
                dprint(f"Considering the following shifts: {shifts_in_interval}")

                # determine what patients are in this interval
                # not a set to prevent randomness
                patients_in_interval = []
                for s_early in early_shifts_in_interval:
                    for p in instance.known_patients_per_shift[s_early, s_update]:
                        if p not in patients_in_interval:
                            patients_in_interval.append(p)
                dprint(f"Patients in current interval: {patients_in_interval}")

                # determine the patients that were also in the previous interval
                patients_in_prev_interval = []
                if k != 0:
                    for p in patients_in_interval:
                        s_adm = instance.patients[p]["shifts"][0]
                        if s_adm < first_shift_of_interval:
                            patients_in_prev_interval.append(p)
                dprint(f"Patients also in previous interval: {patients_in_prev_interval}")


                nurses_in_interval = []
                for s in shifts_in_interval:
                    for n in instance.nurses_per_shift[s]:
                        if n not in nurses_in_interval:
                            nurses_in_interval.append(n)
                dprint(f"Nurses in partition: {nurses_in_interval}")

                first_day_of_interval += interval_length

                #########################################################################################
                # ILP

                env = gp.Env(empty=True)
                if not self.print_ilp_log:
                    env.setParam('OutputFlag', 0)
                env.start()

                model = gp.Model(instance.instance_name, env=env)

                if not self.print_ilp_log:
                    model.Params.LogToConsole = False

                model.Params.Seed = 42 # since each run only once, just keep static seed
                model.Params.TimeLimit = self.time_limit_per_interval[k]

                #################
                #   VARIABLES   #
                #################

                # decision variable for nurses
                valid_x_indices = [(n, r, s) for r in instance.room_ids for s in shifts_in_interval
                                   for n in instance.nurses_per_shift[s]]
                x = model.addVars(valid_x_indices, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="x")
                for (n, r, s) in valid_x_indices:
                    warm_start_index = ("x", n, r, s)
                    if warm_start_index in warm_start_values:
                        x[n, r, s].Start = warm_start_values[warm_start_index]

                # decision variable for patients
                valid_y_indices = [(p, r) for p in patients_in_interval
                                   for r in instance.room_ids]
                y = model.addVars(valid_y_indices, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="y")
                for (p, r) in valid_y_indices:
                    warm_start_index = ("y", p, r)
                    if warm_start_index in warm_start_values:
                        y[p, r].Start = warm_start_values[warm_start_index]

                # Variable for nurse-to-patient assignment
                valid_z_indices = [(n, p, s) for s in shifts_in_interval for n in instance.nurses_per_shift[s]
                                   for p in instance.known_patients_per_shift[s, s_update]]
                z = model.addVars(valid_z_indices, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="z")

                #################
                #  Constraints  #
                #################

                ## Nurse-to-Room assignment
                # Each room assigned to exactly one nurse
                model.addConstrs(gp.quicksum(x[n, r, s] for n in instance.nurses_per_shift[s]) == 1
                                 for r in instance.room_ids for s in shifts_in_interval)

                ## Patient-to-Room assignment
                # Each patient is only assigned to one room
                model.addConstrs(gp.quicksum(y[p, r] for r in instance.room_ids) == 1
                                 for p in patients_in_interval)

                # Each room cannot exceed its capacity
                model.addConstrs(gp.quicksum(y[p, r] for p in instance.known_patients_per_shift[s, s_update])
                                 <= instance.room_capacities[r] for r in instance.room_ids
                                 for s in early_shifts_in_interval)

                # Patients cannot be assigned to incompatible rooms
                model.addConstrs(y[p, r] == 0
                                 for p in patients_in_interval
                                 for r in instance.patients[p]["incompatible_rooms"])

                ## Nurse-to-Patient assignment
                # Nurse is assigned to a patient if they are in the same room
                model.addConstrs(z[n, p, s] >= x[n, r, s] + y[p, r] - 1 for s in shifts_in_interval
                                 for r in instance.room_ids for n in instance.nurses_per_shift[s]
                                 for p in instance.known_patients_per_shift[s, s_update])

                # Only one nurse assigned to each patient in each shift
                model.addConstrs(gp.quicksum(z[n, p, s] for n in instance.nurses_per_shift[s])
                                 == 1 for s in shifts_in_interval
                                 for p in instance.known_patients_per_shift[s, s_update])

                # non-occupant patients must lie in the same room as in the previous interval
                if k != 0:
                    model.addConstrs(y[p,temp_PR_assignment[p,first_shift_of_interval - 3]] == 1 for p in patients_in_prev_interval)

                #################
                #   OBJECTIVE   #
                #################

                obj = gp.LinExpr()

                ## Minimizing number of different nurses
                if instance.weights["Continuity"] != 0:
                    ever_assigned = model.addVars(nurses_in_interval, patients_in_interval,
                                                  lb=0.0, ub=1.0, vtype=GRB.BINARY,
                                                  name="ever_assigned")

                    for n in nurses_in_interval:
                        for p in patients_in_interval:
                            if n in nurses_per_patient[p]:  # if the nurse was used previously
                                model.addConstr(ever_assigned[n, p] == 1)
                            else:
                                # ever_assigned = 1 if at least one time nurse n assigned to patient p
                                model.addConstrs(ever_assigned[n, p] >= z[n, p, s] for s in shifts_in_interval
                                                 if s in instance.patients[p]["shifts"]
                                                 if s in instance.nurses[n]["shifts"])

                                # If z = 0 for all shifts, then ever_assigned = 0
                                model.addConstr(ever_assigned[n, p] <= gp.quicksum(z[n, p, s] for s in shifts_in_interval
                                                                                   if s in instance.patients[p]["shifts"]
                                                                                   if s in instance.nurses[n]["shifts"]))

                    # Add to objective
                    obj += instance.weights["Continuity"] * gp.quicksum(
                        ever_assigned[n, p] for n in nurses_in_interval for p in patients_in_interval)

                ## Minimizing number of gender violations
                if instance.weights["Gender-Mixing"] != 0:
                    # Variable for gender mixing constraint
                    f_in_room = model.addVars(instance.room_ids, early_shifts_in_interval,
                                              lb=0.0, ub=1.0, vtype=GRB.BINARY, name="f_in_room")
                    m_in_room = model.addVars(instance.room_ids, early_shifts_in_interval,
                                              lb=0.0, ub=1.0, vtype=GRB.BINARY, name="m_in_room")
                    gender_vio = model.addVars(instance.room_ids, early_shifts_in_interval,
                                               lb=0.0, ub=1.0, vtype=GRB.BINARY, name="gender_vio")

                    # set m_in_room and f_in_room correctly based on assigned patients
                    model.addConstrs(y[p, r] <= m_in_room[r, s]
                                     for s in early_shifts_in_interval
                                     for p in instance.known_patients_per_shift[s, s_update]
                                     for r in instance.room_ids if instance.patients[p]["gender"] == "A")
                    model.addConstrs(y[p, r] <= f_in_room[r, s]
                                     for s in early_shifts_in_interval
                                     for p in instance.known_patients_per_shift[s, s_update]
                                     for r in instance.room_ids if instance.patients[p]["gender"] == "B")

                    # If both male and female patients, add a violation
                    model.addConstrs(m_in_room[r, s] + f_in_room[r, s] <= 1 + gender_vio[r, s]
                                     for r in instance.room_ids for s in early_shifts_in_interval)

                    obj += instance.weights["Gender-Mixing"] * gp.quicksum(gender_vio[r, s] for r in instance.room_ids
                                                                           for s in early_shifts_in_interval)

                ## Skill requirement
                if instance.weights["Skill Requirements"] != 0:
                    valid_indices = [(p, s) for s in shifts_in_interval
                                     for p in instance.known_patients_per_shift[s, s_update]]
                    skill_vio = model.addVars(valid_indices, lb=0.0, vtype=GRB.CONTINUOUS)

                    model.addConstrs(skill_vio[p, s] >= instance.patients[p]["skill"][s] -
                                     gp.quicksum(instance.nurses[n]["skill"] * z[n, p, s]
                                                 for n in instance.nurses_per_shift[s])
                                     for (p,s) in valid_indices)

                    obj += instance.weights["Skill Requirements"] * gp.quicksum(
                        skill_vio[p, s] for (p,s) in valid_indices)

                ## Minimizing workload violation
                if instance.weights["Workload Violation"] != 0:
                    valid_indices = [(n, s) for s in shifts_in_interval for n in instance.nurses_per_shift[s]]
                    load_vio = model.addVars(valid_indices, lb=0.0, vtype=GRB.CONTINUOUS, name="load_vio")

                    model.addConstrs(gp.quicksum(instance.patients[p]["workload"][s] * z[n, p, s]
                                                 for p in instance.known_patients_per_shift[s, s_update])
                                     <= instance.nurses[n]["max_load"][s] + load_vio[n, s]
                                     for s in shifts_in_interval for n in instance.nurses_per_shift[s])

                    # Add to objective
                    obj += instance.weights["Workload Violation"] * gp.quicksum(
                        load_vio[n, s] for s in shifts_in_interval for n in instance.nurses_per_shift[s])

                # Minimizing workload imbalance per shift
                if instance.weights["Workload Imbalance"] != 0:
                    min_load = model.addVars(shifts_in_interval, lb=0.0, vtype=GRB.CONTINUOUS, name="min_load")
                    max_load = model.addVars(shifts_in_interval, lb=0.0, vtype=GRB.CONTINUOUS, name="max_load")

                    model.addConstrs(min_load[s] <= gp.quicksum(instance.patients[p]["workload"][s]
                                                                / instance.nurses[n]["max_load"][s] * z[n, p, s]
                                                                for p in instance.known_patients_per_shift[s, s_update])
                                     for s in shifts_in_interval for n in instance.nurses_per_shift[s])
                    model.addConstrs(max_load[s] >= gp.quicksum(instance.patients[p]["workload"][s]
                                                                / instance.nurses[n]["max_load"][s] * z[n, p, s]
                                                                for p in instance.known_patients_per_shift[s, s_update])
                                     for s in shifts_in_interval for n in instance.nurses_per_shift[s])

                    # Add to objective
                    obj += instance.weights["Workload Imbalance"] * gp.quicksum(
                        max_load[s] - min_load[s] for s in shifts_in_interval)

                # Minimizing the number of transfers
                if instance.weights.get("Transfers", 0) != 0:
                    if k == 0:
                        trans = model.addVars(occupants, lb=0.0, ub=1.0, vtype=GRB.BINARY,
                                              name="trans")

                        # Minimize transfers from previously fixed PR-assignments
                        model.addConstrs(y[p, prev_room_per_occupant[p]] == 1 - trans[p]
                                         for p in occupants)


                        obj += instance.weights["Transfers"] * gp.quicksum(trans[p] for p in occupants)

                model.setObjective(obj, GRB.MINIMIZE)
                start_time = datetime.datetime.now()
                model.optimize()
                end_time = datetime.datetime.now()

                # Check if the ILP is infeasible or no solution has been found
                if model.Status == GRB.Status.INFEASIBLE or model.Status == GRB.Status.INF_OR_UNBD:
                    dprint(f"Model is infeasible, stopped optimizing.")
                    self.status = "infeasible"
                    return None, solution_history
                if model.SolCount == 0:
                    dprint(f"Could not find solution within given time limit.")
                    self.status = "timeout"
                    return None, solution_history

                # divide the time budget across remaining intervals
                if model.status == GRB.Status.TIME_LIMIT:
                    self.numb_timeouts += 1
                    self.time_spent_per_interval[s_update].append(self.time_limit_per_interval[k])
                    dprint(f'Time limit exceeded')
                else:
                    spent_time = (end_time - start_time).total_seconds()
                    self.time_spent_per_interval[s_update].append(spent_time)
                    remaining_time = self.time_limit_per_interval[k] - spent_time

                    if self.use_proportional_runtime:
                        numb_days_left = sum(partition[k+1:])
                        if numb_days_left != 0:
                            add_time_per_day = remaining_time / numb_days_left
                            for k_0 in range(k+1,K):
                                L_k = partition[k_0]
                                self.time_limit_per_interval[k_0] += L_k * add_time_per_day
                                # print(f"\t\tinterval {k_0} has {L_k} days -> added {L_k * add_time_per_day} seconds")
                    else:
                        numb_of_intervals_left = K - k - 1
                        if numb_of_intervals_left != 0:
                            add_time_per_interval = remaining_time / numb_of_intervals_left
                            for k_0 in range(k+1,K):
                                self.time_limit_per_interval[k_0] += add_time_per_interval
                    # print(f"\tUpdated time limits = {self.time_limit_per_interval}")

                # Update PR assignment
                for (p, r) in valid_y_indices:
                    warm_start_values["y",p, r] = y[p, r].X
                    if abs(y[p, r].X) > 10e-6:
                        for s in shifts_in_interval:
                            if s in instance.early_shifts:
                                if s in instance.patients[p]["shifts"]:
                                    temp_PR_assignment[p, s] = r

                # Update NR assignment
                for (n, r, s) in valid_x_indices:
                    warm_start_values["x",n,r,s] = x[n, r, s].X
                    if abs(x[n, r, s].X) > 10e-6:
                        temp_NR_assignment[n, s].append(r)

                # update nurses_per_patient
                if instance.weights["Continuity"] != 0:
                    for p in patients_in_interval:
                        for n in nurses_in_interval:
                            if abs(ever_assigned[n, p].X) > 10e-6:
                                if n not in nurses_per_patient[p]:
                                    nurses_per_patient[p].append(n)

                solution = (temp_PR_assignment, temp_NR_assignment)
                solution_history[s_update] = solution

            # update fixed PR and NR assignments
            if i == len(instance.schedule_update_shifts) - 1:
                fixed_PR_assignment = temp_PR_assignment
                for (n,s) in temp_NR_assignment:
                    fixed_NR_assignment[n,s] = temp_NR_assignment[n,s]
            else:
                next_shift = instance.schedule_update_shifts[i + 1]
                for s in range(s_update, next_shift, 3): # only early shifts
                    for p in instance.patients_per_shift[s]:
                        fixed_PR_assignment[p, s] = temp_PR_assignment[p,s]
                for s in range(s_update, next_shift):
                    for n in instance.nurses_per_shift[s]:
                        fixed_NR_assignment[n,s] = temp_NR_assignment[n,s]
            dprint(f"\t-> time per interval: {self.time_spent_per_interval[s_update]}")

            self.time_limits_per_s_update[s_update] = self.time_limit_per_interval
            if self.print_extra_info:
                print(f"\t-> time spent: {sum(self.time_spent_per_interval[s_update]):.6f} seconds (out of {time_budget} s)")
                print(f"\t\t-> Time spent per interval: {self.time_spent_per_interval[s_update]}")
                print(f"\t\t-> Time limit per interval: {self.time_limit_per_interval}")

            if self.print_progress:
                print(s_update, end = ", ")
        if self.print_progress:
            print()

        return solution, solution_history

    def get_L_max_per_update_shift(self):
        """Determine the L_max per update shift. This is based on the number of known patients left to schedule"""
        instance = self.instance
        L_max_per_update_shift = dict()
        for s_update in instance.schedule_update_shifts:
            numb_patients_to_schedule = len(instance.schedulable_patients[s_update])
            if numb_patients_to_schedule < 75:
                L_max = self.L_max_per_size['small']
            elif numb_patients_to_schedule < 240:
                L_max = self.L_max_per_size['medium']
            else:
                L_max = self.L_max_per_size['large']
            L_max_per_update_shift[s_update] = L_max
        self.L_max_per_update_shift = L_max_per_update_shift

    def get_best_partition(self):

        # get the correct values for L_max
        self.get_L_max_per_update_shift()

        # determine patient overlap
        if len(self.patient_overlap_per_update_shift) == 0:
            self.get_patient_overlap(False)
        patient_overlap_per_update_shift = self.patient_overlap_per_update_shift

        for s_update in self.instance.schedule_update_shifts:
            # get the interval length
            interval_lengths = self.get_interval_lengths(s_update)

            # determine the number of intervals of each length
            interval_length_counts = dict()
            for i in range(len(interval_lengths)):
                l = interval_lengths[i]
                if l not in interval_length_counts:
                    interval_length_counts[l] = 1
                else:
                    interval_length_counts[l] += 1

            # get all permutations of interval lengths (removing duplicates
            all_partitions = self.get_all_partitions(interval_length_counts, [])

            # determine the best partition
            best_partition = None
            best_overlap = float('inf')
            for partition in all_partitions:
                overlap_value = 0
                d = s_update // 3
                for interval_length in partition[:-1]:
                    d += interval_length
                    overlap_value += patient_overlap_per_update_shift[s_update][d-1]

                if overlap_value < best_overlap:
                    best_partition = partition
                    best_overlap = overlap_value

            # print(f"Best partition shift {s_update}: {best_partition} with overlap {best_overlap}")
            self.K_per_update_shift[s_update] = len(best_partition)
            self.partition_per_update_shift[s_update] = best_partition
            self.partition_overlap_per_update_shift[s_update]  = best_overlap

    def get_patient_overlap(self, show_overlap = False):
        """
        Shows for day d how many patients are present on both day d and d+1.
        returns dictionary patient_overlap_per_update_shift. The keys are update shifts.
        Each item is a dictionary denoting the patient overlap with the known patient in that shift.
        """
        patient_overlap_per_update_shift = dict()
        for s_update in self.instance.schedule_update_shifts:
            patient_overlap = {d: 0 for d in range(self.numberOfDays - 1)}
            for d in patient_overlap:
                s_early = 3 * d
                s_early_next = s_early + 3
                for p in self.instance.known_patients_per_shift[s_early, s_update]:
                    if p in self.instance.known_patients_per_shift[s_early_next, s_update]:
                        patient_overlap[d] += 1
            patient_overlap_per_update_shift[s_update] = patient_overlap

            if show_overlap:
                fig, ax = plt.subplots()
                for d in patient_overlap:
                    ax.bar(3 * d + 1.5, patient_overlap[d], width=3, color="tab:blue", linewidth=1,
                           edgecolor="black")

                plt.ylabel("Number of patients")
                plt.xlabel("Days")
                plt.xticks([3 * x + 1.5 for x in range(self.numberOfDays - 1)],
                           [x + 1 for x in range(self.numberOfDays - 1)])
                plt.xlim([-0.02 * 3 * (self.numberOfDays - 1), 1.02 * 3 * (self.numberOfDays - 1)])
                plt.title(f"Number of patient overlap\n update shift {s_update}")
                plt.show()
        self.patient_overlap_per_update_shift = patient_overlap_per_update_shift

    def get_interval_lengths(self, s_update):
        """
        Determine interval lengths based on the maximum number of days in each interval (L_max).
        The first interval is always set to L_max, while the other intervals are of roughly equal length.
        The length of the intervals decreases such that the first few intervals are longer than later intervals.
        The number of intervals is the amount that the maximum value can fit, rounded up.
        If the number of remaining days is less than L_max, returns only a single interval
        """
        # retrieve values used often
        number_of_days = self.numberOfDays - s_update // 3
        L_max = self.L_max_per_update_shift[s_update]

        if L_max > number_of_days:
            return [number_of_days]

        # get the number of intervals
        if number_of_days % L_max == 0:
            K = number_of_days // L_max
        else:
            K = number_of_days // L_max + 1

        numb_days_left = number_of_days
        intervals_left = K
        interval_lengths = []
        while intervals_left > 0:
            if len(interval_lengths) == 0:  # first day should be of the maximum value
                days_in_interval = L_max
            else:
                if numb_days_left % intervals_left == 0:
                    days_in_interval = numb_days_left // intervals_left
                else:
                    days_in_interval = numb_days_left // intervals_left + 1
            interval_lengths.append(days_in_interval)
            numb_days_left -= days_in_interval
            intervals_left -= 1
        return interval_lengths

    def get_all_partitions(self, interval_length_counts, cur_partition):
        """Recursive function that determines all permutations of interval lengths removing duplicates."""
        if len(interval_length_counts) == 0:
            return [cur_partition]
        else:
            all_partitions = []
            for l in interval_length_counts:
                # copy things to prevent incorrect changes
                new_partition = cur_partition.copy()
                temp_length_counts = interval_length_counts.copy()

                # change
                new_partition.append(l)
                temp_length_counts[l] -= 1
                if temp_length_counts[l] == 0:
                    temp_length_counts.pop(l)

                new_partitions = self.get_all_partitions(temp_length_counts, new_partition)
                all_partitions += new_partitions
            return all_partitions


if __name__ == "__main__":
    from Online_InstanceClass import EmergencyInstance
    from Online_ComputeObjective import compute_objective_emergency

    instance_name = "m01_10_1"
    dataset_name = instance_name_to_dataset(instance_name)
    file_path = "Instances/" + dataset_name + "/" + instance_name + ".json"
    instance = EmergencyInstance(file_path, print_instance_info=False)

    interval_heuristic = OnlineIntervalHeuristic(instance)
    interval_heuristic.L_max_per_size = {'small': 1, 'medium': 1, 'large': 1} # change the value of L_max
    interval_heuristic.get_best_partition()
    interval_heuristic.print_ilp_log = False # shows the ILP log
    interval_heuristic.print_extra_info = False # gives information every time a shift is finished
    interval_heuristic.print_progress = True # print the s_updates that are finished


    start_time = datetime.datetime.now()
    solution, solution_history = interval_heuristic.execute()
    end_time = datetime.datetime.now()
    print(f"Interval heuristic took {(end_time - start_time).total_seconds()} seconds")
    compute_objective_emergency(instance, solution, print_table=True)

