import random
from HelperFunctions import *
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB

class IntervalHeuristic:
    def __init__(self, instance):
        self.instance = instance
        self.numberOfDays = instance.numberOfDays
        self.L_max = 1
        self.patient_overlap = dict()

        self.K = 0
        self.partition = None
        self.partition_overlap = 0

        self.time_budget = 150 * instance.numberOfDays // 7
        self.time_limit_per_interval = []
        self.time_spent_per_interval = []
        self.numb_timeouts = 0 # counts the number of times the ilp was not solved within the time limit
        self.print_ilp_log = False

        self.last_interval_considered = 0
        self.status = None
        # we have three possible status codes:
        #   - "done": the heuristic has finished successfully
        #   - "infeasible": the heuristic terminated early because a previous decision led to infeasible ILP
        #   - "timeout": the ILP was not able to find a solution in one interval due to time limit

        self.get_best_partition()

    def execute(self):
        # time budget is equally spread across the intervals
        self.time_limit_per_interval = [self.time_budget / self.K for k in range(self.K)]
        self.time_spent_per_interval = []
        self.numb_timeouts = 0

        # retrieve as it is often used
        instance = self.instance

        PR_assignment = dict()
        NR_assignment = {(n, s): [] for n in self.instance.nurse_ids for s in self.instance.nurses[n]["shifts"]}

        nurses_per_patient = {p: [] for p in self.instance.patient_ids}

        # add the current occupants to the correct room
        for p in instance.occupant_ids:
            PR_assignment[p] = instance.patients[p]["prev_room"]

        first_day_of_part = 0
        for k in range(self.K):
            self.last_interval_considered = k
            part = self.partition[k]

            days_in_part = list(range(first_day_of_part, first_day_of_part + part))
            early_shifts_in_part = [3 * day for day in days_in_part]
            shifts_in_part = list(range(3 * first_day_of_part, 3 * (first_day_of_part + part)))
            dprint(f"\nConsidering the following days: {days_in_part}")
            dprint(f"Considering the following shifts: {shifts_in_part}")

            # determine what patients are in this interval
            # not a set to prevent randomness
            patients_in_part = []
            for d in days_in_part:
                s_early = 3 * d
                for p in instance.patients_per_shift[s_early]:
                    if p not in patients_in_part:
                        patients_in_part.append(p)
            dprint(f"Patients in partition: {patients_in_part}")

            nurses_in_part = []
            for s in shifts_in_part:
                for n in instance.nurses_per_shift[s]:
                    if n not in nurses_in_part:
                        nurses_in_part.append(n)
            dprint(f"Nurses in partition: {nurses_in_part}")

            first_day_of_part += part

            #########################################################################################
            # ILP

            env = gp.Env(empty=True)
            if not self.print_ilp_log:
                env.setParam('OutputFlag', 0)
            env.start()

            model = gp.Model(instance.instance_name, env=env)
            # model.Params.NoRelHeurTime = 0

            if not self.print_ilp_log:
                model.Params.LogToConsole = False

            model.Params.Seed = random.randint(1, 10000)
            model.Params.TimeLimit = self.time_limit_per_interval[k]

            #################
            #   VARIABLES   #
            #################

            # decision variable for nurses
            valid_x_indices = [(n, r, s) for r in instance.room_ids for s in shifts_in_part
                               for n in instance.nurses_per_shift[s]]
            x = model.addVars(valid_x_indices, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="x")

            # decision variable for patients
            valid_y_indices = [(p, r) for p in patients_in_part for r in instance.room_ids]
            y = model.addVars(valid_y_indices, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="y")

            # Variable for nurse-to-patient assignment
            valid_z_indices = [(n, p, s) for s in shifts_in_part for n in instance.nurses_per_shift[s]
                               for p in instance.patients_per_shift[s]]
            z = model.addVars(valid_z_indices, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="z")

            #################
            #  Constraints  #
            #################

            ## Nurse-to-Room assignment
            # Each room assigned to exactly one nurse
            model.addConstrs(gp.quicksum(x[n, r, s] for n in instance.nurses_per_shift[s]) == 1
                             for r in instance.room_ids for s in shifts_in_part)

            ## Patient-to-Room assignment
            # Each patient is only assigned to one room
            model.addConstrs(gp.quicksum(y[p, r] for r in instance.room_ids) == 1 for p in patients_in_part)

            # Each room cannot exceed its capacity
            model.addConstrs(gp.quicksum(y[p, r] for p in instance.patients_per_shift[s])
                             <= instance.room_capacities[r] for r in instance.room_ids for s in early_shifts_in_part)

            # Patients cannot be assigned to incompatible rooms
            model.addConstrs(y[p, r] == 0 for p in patients_in_part for r in instance.patients[p]["incompatible_rooms"])

            # Keep previously assigned patients in the same room as before
            model.addConstrs(y[p, PR_assignment[p]] == 1 for p in patients_in_part
                             if p in PR_assignment)

            ## Nurse-to-Patient assignment
            # Nurse is assigned to a patient if they are in the same room
            model.addConstrs(z[n, p, s] >= x[n, r, s] + y[p, r] - 1 for s in shifts_in_part
                             for r in instance.room_ids for n in instance.nurses_per_shift[s]
                             for p in instance.patients_per_shift[s])

            # Only one nurse assigned to each patient in each shift
            model.addConstrs(gp.quicksum(z[n, p, s] for n in instance.nurses_per_shift[s])
                             == 1 for s in shifts_in_part for p in instance.patients_per_shift[s])

            #################
            #   OBJECTIVE   #
            #################

            obj = gp.LinExpr()

            ## Minimizing number of different nurses
            if instance.weights["Continuity"] != 0:
                ever_assigned = model.addVars(nurses_in_part, patients_in_part, lb=0.0, ub=1.0, vtype=GRB.BINARY,
                                              name="ever_assigned")

                for p in patients_in_part:
                    for n in nurses_in_part:
                        if n in nurses_per_patient[p]:  # if the nurse was used previously
                            model.addConstr(ever_assigned[n, p] == 1)
                        else:
                            # ever_assigned = 1 if at least one time nurse n assigned to patient p
                            model.addConstrs(ever_assigned[n, p] >= z[n, p, s] for s in shifts_in_part
                                             if s in instance.patients[p]["shifts"]
                                             if s in instance.nurses[n]["shifts"])

                            # If z = 0 for all shifts, then ever_assigned = 0
                            model.addConstr(ever_assigned[n, p] <= gp.quicksum(z[n, p, s] for s in shifts_in_part
                                                                               if s in instance.patients[p]["shifts"]
                                                                               if s in instance.nurses[n]["shifts"]))

                # Add to objective
                obj += instance.weights["Continuity"] * gp.quicksum(
                    ever_assigned[n, p] for n in nurses_in_part for p in patients_in_part)

            ## Minimizing number of gender violations
            if instance.weights["Gender-Mixing"] != 0:
                # Variable for gender mixing constraint
                f_in_room = model.addVars(instance.room_ids, early_shifts_in_part, lb=0.0, ub=1.0, vtype=GRB.BINARY,
                                          name="f_in_room")
                m_in_room = model.addVars(instance.room_ids, early_shifts_in_part, lb=0.0, ub=1.0, vtype=GRB.BINARY,
                                          name="m_in_room")
                gender_vio = model.addVars(instance.room_ids, early_shifts_in_part, lb=0.0, ub=1.0, vtype=GRB.BINARY,
                                           name="gender_vio")

                # set m_in_room and f_in_room correctly based on assigned patients
                model.addConstrs(
                    y[p, r] <= m_in_room[r, s] for s in early_shifts_in_part for p in instance.patients_per_shift[s]
                    for r in instance.room_ids if instance.patients[p]["gender"] == "A")
                model.addConstrs(
                    y[p, r] <= f_in_room[r, s] for s in early_shifts_in_part for p in instance.patients_per_shift[s]
                    for r in instance.room_ids if instance.patients[p]["gender"] == "B")

                # If both male and female patients, add a violation
                model.addConstrs(m_in_room[r, s] + f_in_room[r, s] <= 1 + gender_vio[r, s] for r in instance.room_ids
                                 for s in early_shifts_in_part)

                obj += instance.weights["Gender-Mixing"] * gp.quicksum(gender_vio[r, s] for r in instance.room_ids
                                                                       for s in early_shifts_in_part)

            ## Skill requirement
            if instance.weights["Skill Requirements"] != 0:
                valid_indices = [(p, s) for s in shifts_in_part for p in instance.patients_per_shift[s]]
                skill_vio = model.addVars(valid_indices, lb=0.0, vtype=GRB.CONTINUOUS)

                model.addConstrs(skill_vio[p, s] >= instance.patients[p]["skill"][s] -
                                 gp.quicksum(instance.nurses[n]["skill"] * z[n, p, s]
                                             for n in instance.nurses_per_shift[s])
                                 for s in shifts_in_part for p in instance.patients_per_shift[s])

                obj += instance.weights["Skill Requirements"] * gp.quicksum(
                    skill_vio[p, s] for s in shifts_in_part for p in instance.patients_per_shift[s])

            ## Minimizing workload violation
            if instance.weights["Workload Violation"] != 0:
                valid_indices = [(n, s) for s in shifts_in_part for n in instance.nurses_per_shift[s]]
                load_vio = model.addVars(valid_indices, lb=0.0, vtype=GRB.CONTINUOUS, name="load_vio")

                model.addConstrs(gp.quicksum(instance.patients[p]["workload"][s] * z[n, p, s]
                                             for p in instance.patients_per_shift[s])
                                 <= instance.nurses[n]["max_load"][s] + load_vio[n, s]
                                 for s in shifts_in_part for n in instance.nurses_per_shift[s])

                # Add to objective
                obj += instance.weights["Workload Violation"] * gp.quicksum(
                    load_vio[n, s] for s in shifts_in_part for n in instance.nurses_per_shift[s])

            # Minimizing workload imbalance per shift
            if instance.weights["Workload Imbalance"] != 0:
                min_load = model.addVars(shifts_in_part, lb=0.0, vtype=GRB.CONTINUOUS, name="min_load")
                max_load = model.addVars(shifts_in_part, lb=0.0, vtype=GRB.CONTINUOUS, name="max_load")

                model.addConstrs(min_load[s] <= gp.quicksum(instance.patients[p]["workload"][s]
                                                            / instance.nurses[n]["max_load"][s] * z[n, p, s]
                                                            for p in instance.patients_per_shift[s])
                                 for s in shifts_in_part for n in instance.nurses_per_shift[s])
                model.addConstrs(max_load[s] >= gp.quicksum(instance.patients[p]["workload"][s]
                                                            / instance.nurses[n]["max_load"][s] * z[n, p, s]
                                                            for p in instance.patients_per_shift[s])
                                 for s in shifts_in_part for n in instance.nurses_per_shift[s])

                # Add to objective
                obj += instance.weights["Workload Imbalance"] * gp.quicksum(
                    max_load[s] - min_load[s] for s in shifts_in_part)

            model.setObjective(obj, GRB.MINIMIZE)
            start_time = datetime.datetime.now()
            model.optimize()
            end_time = datetime.datetime.now()

            # Check if the ILP is infeasible or no solution has been found
            if model.Status == GRB.Status.INFEASIBLE or model.Status == GRB.Status.INF_OR_UNBD:
                dprint(f"Model is infeasible, stopped optimizing.")
                self.status = "infeasible"
                return None
            if model.SolCount == 0:
                dprint(f"Could not find solution within given time limit.")
                self.status = "timeout"
                return None

            # divide the time budget across remaining intervals
            if model.status == GRB.Status.TIME_LIMIT:
                self.numb_timeouts += 1
                self.time_spent_per_interval.append(self.time_limit_per_interval[k])
                dprint(f'Time limit exceeded')
            else:
                spent_time = (end_time - start_time).total_seconds()
                self.time_spent_per_interval.append(spent_time)
                remaining_time = self.time_limit_per_interval[k] - spent_time
                numb_of_intervals_left = self.K - k - 1
                dprint(f"Time limit met. \n"
                      f"\tTime spent: {spent_time} \n"
                      f"\tremaining time = {remaining_time}\n"
                      f"\tTime will be divided over {numb_of_intervals_left} intervals")

                if numb_of_intervals_left != 0:
                    add_time_per_interval = remaining_time / numb_of_intervals_left
                    for k_0 in range(k+1,self.K):
                        self.time_limit_per_interval[k_0] += add_time_per_interval
                dprint(f"\tUpdated time limits = {self.time_limit_per_interval}")


            # Update NR assignment
            for (n, r, s) in valid_x_indices:
                if abs(x[n, r, s].X) > 10e-6:
                    NR_assignment[n, s].append(r)

            # Update PR assignment
            for (p, r) in valid_y_indices:
                if abs(y[p, r].X) > 10e-6:
                    if p not in PR_assignment:
                        PR_assignment[p] = r

            # update nurses_per_patient
            if instance.weights["Continuity"] != 0:
                for p in patients_in_part:
                    for n in nurses_in_part:
                        if abs(ever_assigned[n, p].X) > 10e-6:
                            if n not in nurses_per_patient[p]:
                                nurses_per_patient[p].append(n)

        solution = (PR_assignment, NR_assignment)
        self.status = "done"
        return solution

    def get_best_partition(self):
        interval_lengths = self.get_interval_lengths()

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

        # determine patient overlap
        if len(self.patient_overlap) == 0:
            self.get_patient_overlap()
        patient_overlap = self.patient_overlap

        # determine the best partition
        best_partition = None
        best_overlap = float('inf')
        for partition in all_partitions:
            overlap_value = 0
            d = 0
            for interval_length in partition[:-1]:
                d += interval_length
                overlap_value += patient_overlap[d-1]
            if overlap_value < best_overlap:
                best_partition = partition
                best_overlap = overlap_value
        self.K = len(best_partition)
        self.partition = best_partition
        self.partition_overlap = best_overlap

    def get_patient_overlap(self, show_overlap=False):
        """Shows for day d how many patients are present on both day d and d+1"""
        patient_overlap = {d: 0 for d in range(self.numberOfDays - 1)}
        for d in patient_overlap:
            s_early = 3 * d
            s_early_next = s_early + 3
            for p in self.instance.patients_per_shift[s_early]:
                if p in self.instance.patients_per_shift[s_early_next]:
                    patient_overlap[d] += 1
        self.patient_overlap = patient_overlap

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
            plt.title("Number of patient overlap")
            plt.show()

    def get_interval_lengths(self):
        """
        Determine interval lengths based on the maximum number of days in each interval (L_max).
        The first interval is always set to L_max, while the other intervals are of roughly equal length.
        The length of the intervals decreases such that the first few intervals are longer than later intervals.
        The number of intervals is the amount that the maximum value can fit, rounded up.
        """
        # retrieve values used often
        number_of_days = self.numberOfDays
        L_max = self.L_max

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

    def set_Lmax(self, L_max):
        if L_max != self.L_max:
            if L_max > self.numberOfDays:
                L_max = self.numberOfDays
            self.L_max = L_max
            self.get_best_partition()
            dprint(f"New partition = {self.partition} with overlap = {self.partition_overlap}")

    def is_within_budget(self):
        total_time_spent = sum(self.time_spent_per_interval)
        if total_time_spent >= self.time_budget - 10e-6: # preventing rounding errors
            return False
        else:
            return True

if __name__ == "__main__":
    from ComputeObjective import compute_objective
    from InstanceClass import Instance
    random.seed(42)

    instance_name = "test03"
    dataset_name = instance_name_to_dataset(instance_name)
    file_path = "Instances/" + dataset_name + "/" + instance_name + ".json"
    instance = Instance(file_path, print_instance_info=True)

    interval_heuristic = IntervalHeuristic(instance)
    interval_heuristic.set_Lmax(1)
    solution = interval_heuristic.execute()
    solution_attributes = get_solution_attributes(instance, solution)
    obj_value, obj_components = compute_objective(instance, solution_attributes, True)

