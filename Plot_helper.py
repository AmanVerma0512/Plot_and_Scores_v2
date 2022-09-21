#!/usr/bin/env python
# coding: utf-8

# In[5]:


import math
import numpy as np
import matplotlib.pyplot as plt
import json
import statistics
import streamlit as st
from scipy.signal import find_peaks


print_flag = 0
plot_flag = 1

peak_flag = True
peak_and_tempo_flag = True
jitter_flag = True
sudden_release_flag = True
smooth_blips_flag = True

def getStartEnd(y):
    # returns first non zero element index and last non zero index
    start = 0
    end = 0
    for i in range(len(y)):
        if y[i] != 0:
            end = i

    for i in range(len(y)):
        if y[len(y) - i - 1] != 0:
            start = len(y) - i
    return start, end


class Augment:
    # Add break
    def stop(self, y):
        fs = FormScore(y)
        sud = fs.sudden_release()
        _, rep_t, _ = fs.peak_and_tempo()

        a = sud[int(len(sud) / 2)]
        y2 = y[:a] + [0 for i in range(int(rep_t))] + y[a:]
        return y2

    def smoothen(self, y, window_size=5):
        avg = []
        for i in range(len(y)):
            avg.append(np.sum(y[max(0, i - window_size):i]) / window_size)
        return avg


class FormScore:
    def __init__(self, y, plot_dir=None):
        self.y = y
        #         print(self.y)
        self.plot_flag=1
        if plot_dir is not None:
            self.plot_flag=1
            self.plot_dir=plot_dir
        self.h_params = {"global_score": 25,
                         "bwt": 60,
                         "gender": "men's",
                         "exercise_mode": "Equipped Powerlifting",
                         "l0": 1,
                         "l1": 1,
                         "l2": 1,
                         "l3": 1,
                         "discount": 0.9,
                         "peaks": {
                             "sz": 12,
                             "max_win": 100
                         },
                         "sudden_release": {
                             "max_to_fall_ratio": 0.4,
                             "fall_time": 4
                         },
                         "mode": {
                             "sz": 12
                         },
                         "jitter": {
                             "window_size": 4,
                             "delta": 2,
                             "t0": 2,
                             "x_dist_rel": 0.2
                         },
                         "smooth_blips": {
                             "sm": 2
                         },

                         "print": 0,
                         "plot": 1,
                         "log_dir": "D:/Forge/Forge/jupyter/formscore-log/",
                         }
        self.print_flag = self.h_params["print"]
        print_flag = self.h_params["print"]
        # self.plot_flag = self.h_params["plot"]
        # plot_flag = self.h_params["plot"]
        self.start, self.end = getStartEnd(self.y)

    def plot_y(self):
        if self.plot_flag:
            plt.rcParams["figure.figsize"] = (15, 4)
            plt.plot(range(len(self.y)), self.y)
            plt.show()

    def peaks(self):
        # peak detection based on class intervals
        sz = self.h_params["peaks"]["sz"]
        max_win = self.h_params["peaks"]["max_win"]
        global peak_flag
        y = self.y
        start = 0
        end = 0
        for i in range(len(y)):
            if y[i] != 0:
                end = i

        for i in range(len(y)):
            if y[len(y) - i - 1] != 0:
                start = len(y) - i
        if type(y)==tuple:
            y=y[0]
        win = max(y) / sz
        curr = win / 2
        flag = 0
        a = 0
        b = 0
        crests = []
        for i in range(start, end):
            curr = y[i]
            win = max(y[max(i - max_win, 0):i]) / sz
            if curr > max(y[max(i - max_win, 0):i]) - win and flag == 0:
                flag = 1
                a = i
            if curr < max(y[max(i - max_win, 0):i]) - 2 * win and flag == 1:
                flag = 0
                b = i
                p = a
                l = 0
                for k in range(a, b + 1):
                    if y[k] > l:
                        l = y[k]
                        p = k
                crests.append(p)
        if peak_flag:
            fig, ax = plt.subplots(figsize=(15,4))
            ax.plot(range(len(y)), y)
            ax.set_title("Peaks")
            roots = crests
            vals = range(len(y))
            ax.plot(crests, [y[i] for i in crests], ls="", marker="o", label="points")
            st.pyplot(fig)
            peak_flag = False
        return crests

    def peak_and_tempo(self):
        # return array, avg, standard deviation
        global peak_and_tempo_flag
        start = 0
        end = 0
        y = self.y
        for i in range(len(y)):
            if y[i] != 0:
                end = i

        for i in range(len(y)):
            if y[len(y) - i - 1] != 0:
                start = len(y) - i
        if self.print_flag == 1:
            print(start, end)
        xp = self.peaks()
        xp = [int(i) for i in xp]

        for p in xp:
            if p < 0.3 * max(y):
                xp.remove(p)
                if self.print_flag == 1:
                    print("***")

        curr_tempo = []
        for i in range(1, len(xp)):
            curr_tempo.append(xp[i] - xp[i - 1])
        if self.print_flag == 1:
            print("Tempo across reps:", curr_tempo, " \nAverage Tempo:", sum(curr_tempo) / len(curr_tempo))
            print("Variance in tempo: ", np.var(curr_tempo))
        if peak_and_tempo_flag:
            fig, ax = plt.subplots(figsize=(15,4))
            ax.plot(range(len(y)), y)
            ax.set_title("Peaks and Tempo")
            roots = xp
            vals = range(len(y))
            ax.plot(xp, [y[i] for i in xp], ls="", marker="o", label="points")
            st.pyplot(fig)
            peak_and_tempo_flag = False

        return curr_tempo, sum(curr_tempo) / len(curr_tempo) if len(curr_tempo) else 0, np.var(curr_tempo)

    def sudden_release(self):
        # detecting using high negative slope and low value
        global sudden_release_flag

        max_to_fall_ratio = self.h_params["sudden_release"]["max_to_fall_ratio"]
        fall_time = self.h_params["sudden_release"]["fall_time"]
        if self.plot_flag:
            plt.rcParams["figure.figsize"] = (15, 4)
        start = 0
        end = 0
        y = self.y
        for i in range(len(y)):
            if y[i] != 0:
                end = i

        for i in range(len(y)):
            if y[len(y) - i - 1] != 0:
                start = len(y) - i
        if self.print_flag == 1:
            print(start, end)
        if type(y)==tuple:
            y=y[0]
        m = max(y)
        if self.print_flag == 1:
            print(m)

        delta = m * max_to_fall_ratio
        t0 = fall_time
        sud = []
        for i in range(len(y)):
            j = 0
            while j <= t0 and y[i - j] >= y[i]:
                j += 1

            if j != 0 and y[i - j + 1] - y[i] > delta:
                sud.append(i)
        if sudden_release_flag:
            fig, ax = plt.subplots(figsize=(15,4))
            ax.plot(range(len(y)), y)
            ax.set_title("Sudden Release")
            roots = sud
            vals = range(len(y))

            mark = [vals.index(i) for i in roots]
            ax.plot(roots, [y[i] for i in mark], ls="", marker="o", label="points")
            st.pyplot(fig)
            sudden_release_flag = False
        return sud

    def mode(self):
        # class interval based node
        sz = self.h_params["mode"]["sz"]
        y = self.y
        win = max(y) / sz
        curr = win / 2
        d = {}
        for i in range(sz):
            curr += win
            d[curr] = 0
        for val in y:
            key = min(d.keys(), key=lambda x: abs(x - val))
            d[key] += 1
        max_val = 0
        ans = 0
        for key in d.keys():
            if key == min(d.keys()):
                continue
            if d[key] > max_val:
                max_val = d[key]
                ans = key
        if self.print_flag == 1:
            print(d)
        return ans

    def jitter(self, window_size=4, delta=2, t0=2, x_dist_rel=0.2):
        # detecting using moving average intersections
        window_size = self.h_params["jitter"]["window_size"]
        delta = self.h_params["jitter"]["delta"]
        t0 = self.h_params["jitter"]["t0"]
        x_dist_rel = self.h_params["jitter"]["x_dist_rel"]

        global jitter_flag

        m = self.mode()
        y = self.y
        if self.print_flag == 1:
            print("Mode: ", m)
        moving_averages = []
        jitter = []

        for i in range(len(y)):
            window_average = np.sum(y[max(0, i - window_size):i]) / window_size
            moving_averages.append(window_average)
            k = 0
            cross = []
            for j in range(t0):
                if y[i - j] > window_average and y[i - j - 1] < window_average or y[i - j] < window_average and y[
                    i - j - 1] > window_average and abs(y[i] - m) < x_dist_rel * max(y):
                    k += 1
                    cross.append(i - j)
            if k >= delta and abs(y[i] - m) < x_dist_rel * max(y):
                jitter.append(int(sum(cross) / len(cross)))
        if self.print_flag == 1:
            print(len(jitter))

        start, end = getStartEnd(y)
        if self.print_flag == 1:
            print(start, end)
        xp = self.peaks()
        xp = [int(i) for i in xp]

        dj = {}
        for p in xp:
            dj[p] = []

        xp.insert(0, 0)
        xp.append(len(y) - 1)

        # merging within a rep
        for j in jitter:
            for i in range(len(xp) - 1):
                l = xp[i]
                r = xp[i + 1]
                if i < (len(xp) - 1) and j >= l and j <= r:
                    dj[l].append(j)

        fj = []
        for key in dj.keys():
            if len(dj[key]) > 0:
                fj.append(int(sum(dj[key]) / len(dj[key])))
#         if self.print_flag == 1:
#             pprint.pprint(dj)

        for j in jitter:
            if y[j] < max(y) / 12:
                jitter.remove(j)

        if jitter_flag:
            fig, ax = plt.subplots(figsize=(15,4))
            ax.plot(range(len(y)), y)
            ax.plot(range(len(y)), moving_averages)
            ax.set_title("Jitter")
            roots = jitter
            vals = range(len(y))
            mark = [vals.index(i) for i in roots]

            valsx = range(len(y))
            markx = [valsx.index(i) for i in xp]
            ax.plot(roots, [y[i] for i in mark], ls="", marker="o", label="points")

            roots = xp
            vals = range(len(y))
            ax.plot(xp, [y[i] for i in xp], ls="", marker="x", label="points")
            st.pyplot(fig)
            jitter_flag = False
        
        return jitter

    def smooth_blips(self, sm=2):
        # jitter detection using scipy peak detection ratio
        global smooth_blips_flag
        y = self.y
        sm = self.h_params["smooth_blips"]["sm"]
        y = self.y
        y = Augment().smoothen(y, sm)
        prominent_peaks, _ = find_peaks(y, prominence=10, width=2)

        y_max = statistics.mean([y[i] for i in prominent_peaks])
        small_peaks, _ = find_peaks(y, height=[0.6 * y_max, 0.8 * y_max], width=2)
        if self.plot_flag:
            plt.plot(range(len(y)), y)
        roots = small_peaks
        vals = range(len(y))
        if self.plot_flag:

            plt.plot(small_peaks, [y[i] for i in small_peaks], ls="", marker="o", label="points")

        roots = prominent_peaks
        vals = range(len(y))
        if smooth_blips_flag:
            fig, ax = plt.subplots(figsize=(15,4))
            ax.plot(range(len(y)), y)
            ax.set_title("Smooth Blips")
            roots = small_peaks
            vals = range(len(y))
            ax.plot(small_peaks, [y[i] for i in small_peaks], ls="", marker="o", label="points")
            roots = prominent_peaks
            vals = range(len(y))
            ax.plot(prominent_peaks, [y[i] for i in prominent_peaks], ls="", marker="x", label="points")
            st.pyplot(fig)
            smooth_blips_flag = False

        if self.print_flag == 1:
            print("Prominent Peaks (x): ", len(prominent_peaks), "\nSmall peaks (o): ", len(small_peaks),
                  "\nScore: ", len(prominent_peaks) / (len(small_peaks) + len(prominent_peaks)))
        return len(prominent_peaks) / (len(small_peaks) + len(prominent_peaks))

    def area_stamina(self):
        # stamina score using area
        ref = int(max(self.y) * 1.4)
        start, end = getStartEnd(self.y)
        total_power = sum(self.y[start:end])
        ref_ls = []
        for i in range(len(self.y)):
            if i <= end and i >= start:
                ref_ls.append(ref)
            else:
                ref_ls.append(0)
        ideal_power = ref * (end - start)
        score = total_power / ideal_power

        # if plot_flag:
        plt.plot(self.y)
        plt.plot(ref_ls)
        # if print_flag:
        print("Stamina Score: ", score)
        plt.show()
        return score

    def dummy_area_stamina2(self):
        ref = int(max(self.y) * 1.4)
        start, end = getStartEnd(self.y)

        y = self.y + self.y[start:end]
        start, end = getStartEnd(y)
        ref_ls = []
        for i in range(len(y)):
            if i <= end and i >= start:
                ref_ls.append(ref)
                y[i] += 5
            else:
                ref_ls.append(0)

        total_power = sum(y[start:end])
        ideal_power = ref * (end - start)
        score = total_power / ideal_power

        # if plot_flag:
        plt.plot(y)
        plt.plot(ref_ls)
        # if print_flag:
        #         print("Stamina Score: ", score)
        plt.show()
        return score

    def area(self):
        start, end = getStartEnd(self.y)
        return sum(self.y[start:end])

    def area_till_reference_time(self, ref):
        start, _ = getStartEnd(self.y)
        return sum(self.y[start:ref])

    def rings_stamina_v0(self, power_ref=0.8, time_ref=200):
        #  v0 of stamaina calculation using punishing factors
        start, end = getStartEnd(self.y)
        xp = self.peaks()
        xp = [self.y[int(i)] for i in xp]
        ref_y = statistics.mean(xp[:min(3, len(xp))])
        ref_y *= power_ref
        ref = (end - start) * ref_y
        ref2 = time_ref * ref_y
        area = self.area()
        pow = min(1, (ref2 - area) / ref)
        ref_plt = []
        ref2_plt = []
        for i in range(len(self.y)):
            if i <= end and i >= start:
                ref_plt.append(ref_y)
            else:
                ref_plt.append(0)
            if i <= start + time_ref and i >= start:
                ref2_plt.append(ref_y)
            else:
                ref2_plt.append(0)

        t = end - start
        t = min(1, (t) / time_ref)

        score = max(0, min(1 - (pow * t), 1)) * 100
        return score

    #   return 1/(pow*i)

    def area(self):
        start, end = getStartEnd(self.y)
        return sum(self.y[start:end])

    def area_till_reference_time(self, ref):
        start, _ = getStartEnd(self.y)
        return sum(self.y[start:ref])

    def rings_stamina_v2(self, power_ref=0.8, time_ref=250):
        # calculates ring stamina using reference power and time punishing factors
        start, end = getStartEnd(self.y)
        xp = self.peaks()
        xp = [self.y[int(i)] for i in xp]
        ref_y = statistics.mean(xp[:min(3, len(xp))])
        m = self.mode()
        ref_y -= m
        ref_y *= power_ref
        t0 = end - start
        if t0 > time_ref:
            ref = ref_y * (time_ref) * time_ref / (t0 - time_ref)
        else:
            ref = (end - start) * ref_y

        ref2 = time_ref * ref_y
        area = self.area() - m * (end - start)
        pow = min(1, (ref2 - area) / ref)
        ref_plt = []
        ref2_plt = []
        for i in range(len(self.y)):
            if i <= end and i >= start:
                ref_plt.append(ref_y)
            else:
                ref_plt.append(0)
            if i <= start + time_ref and i >= start:
                ref2_plt.append(ref_y)
            else:
                ref2_plt.append(0)

        t = end - start
        t = min(1, (t) / time_ref)
        print("Time parameters: ", end - start, time_ref)
        print("Reference power : ", ref)
        print("Punishing Power factor: ", pow)
        print("Punishing Time factor: ", t)
        plt.plot(self.y, label="y")
        plt.plot(ref_plt, label="ref")
        plt.plot(ref2_plt, label="ref2")
        plt.legend()
        plt.show()
        score = max(0, min(1 - (pow * t), 1)) * 100
        return score

    def rings_stamina_v2(self, power_ref=0.8, time_ref=250):
        # calculates ring stamina using reference power and time punishing factors
        start, end = getStartEnd(self.y)
        xp = self.peaks()
        xp = [self.y[int(i)] for i in xp]
        ref_y = statistics.mean(xp[:min(3, len(xp))])
        m = self.mode()
        ref_y -= m
        ref_y *= power_ref
        t0 = end - start
        if t0 > time_ref:
            ref = ref_y * (time_ref) * time_ref / (t0 - time_ref)
        else:
            ref = (end - start) * ref_y

        ref2 = time_ref * ref_y
        area = self.area() - m * (end - start)
        pow = min(1, (ref2 - area) / ref)
        ref_plt = []
        ref2_plt = []
        for i in range(len(self.y)):
            if i <= end and i >= start:
                ref_plt.append(ref_y)
            else:
                ref_plt.append(0)
            if i <= start + time_ref and i >= start:
                ref2_plt.append(ref_y)
            else:
                ref2_plt.append(0)

        t = end - start
        t = min(1, (t) / time_ref)
        score = max(0, min(1 - (pow * t), 1)) * 100
        return score

    def rings_stamina_v1(self, power_ref=0.8, time_ref=250):
        # calculates ring stamina using reference power and time punishing factors
        start, end = getStartEnd(self.y)
        xp = self.peaks()
        xp = [self.y[int(i)] for i in xp]
        ref_y = statistics.mean(xp[:min(3, len(xp))])
        m = self.mode()
        ref_y -= m
        ref_y *= power_ref
        ref = (end - start) * ref_y
        ref2 = time_ref * ref_y
        area = self.area() - m * (end - start)
        pow = min(1, (ref2 - area) / ref)
        ref_plt = []
        ref2_plt = []
        for i in range(len(self.y)):
            if i <= end and i >= start:
                ref_plt.append(ref_y)
            else:
                ref_plt.append(0)
            if i <= start + time_ref and i >= start:
                ref2_plt.append(ref_y)
            else:
                ref2_plt.append(0)

        t = end - start
        t = min(1, (t) / time_ref)
        score = max(0, min(1 - (pow * t), 1)) * 100
        return score

    def power(self, bwt=60, gender="men's", exercise_mode="Equipped Powerlifting"):
        if self.h_params["bwt"]:
            bwt = self.h_params["bwt"]
        if self.h_params["gender"]:
            gender = self.h_params["gender"]
        if self.h_params["mode"]:
            exercise_mode = self.h_params["exercise_mode"]

        params = {
            "men's":
                {"Equipped Powerlifting":
                     {"A": 1236.25115,
                      "B": 1449.21864,
                      "C": 0.01644},
                 "Classic Powerlifting":
                     {"A": 1199.72839,
                      "B": 1025.18162,
                      "C": 0.00921},
                 "Equipped Bench Press":
                     {"A": 381.22073,
                      "B": 733.79378,
                      "C": 0.02398},
                 "Classic Bench Press":
                     {"A": 320.98041,
                      "B": 281.40258,
                      "C": 0.01008}},
            "women's":
                {"Equipped Powerlifting":
                     {"A": 758.63878,
                      "B": 949.31382,
                      "C": 0.02435},
                 "Classic Powerlifting":
                     {"A": 610.32796,
                      "B": 1045.59282,
                      "C": 0.03048},
                 "Equipped Bench Press":
                     {"A": 221.82209,
                      "B": 357.00377,
                      "C": 0.02937},
                 "Classic Bench Press":
                     {"A": 142.40398,
                      "B": 442.52671,
                      "C": 0.04724}},

        }
        xp = self.peaks()
        d = {}
        for i in xp:
            d[i] = self.y[i]
        d = {k: v for k, v in sorted(d.items(), key=lambda item: -item[1])}
        c = min(3, len(xp))
        o = 0
        power = 0
        for key in list(d.keys()):
            power += d[key]
            o += 1
            if (o > c):
                break
        power /= c
        coeff = 100 / (params[gender][exercise_mode]["A"] - params[gender][exercise_mode]["B"] * (
                    2.718281828459 ** (-1 * params[gender][exercise_mode]["C"] * bwt)))
        return power, power * coeff


class Policy:
    # flags one rep based on threshold, and returns fraction of reps which do not cross the threshold
    def rep_and_threshold(y, detected, threshold):
        start = 0
        end = 0
        for i in range(len(y)):
            if y[i] != 0:
                end = i

        for i in range(len(y)):
            if y[len(y) - i - 1] != 0:
                start = len(y) - i
        if print_flag == 1:
            print(start, end)
        xp = FormScore(y).peaks()
        xp = [int(i) for i in xp]

        dj = {}
        for p in xp:
            dj[p] = []

        xp.insert(0, 0)
        xp.append(len(y) - 1)
#         if print_flag == 1:
#             pprint.pprint(dj)

        for p in xp:
            if p >= start or p >= end:
                xp.remove(p)

        # merging within a rep
        for j in detected:
            for i in range(1, len(xp) - 1):
                l = xp[i]
                r = xp[i + 1]
                if i < (len(xp) - 1) and j >= l and j <= r:
                    dj[l].append(j)

        fj = 0
        for key in dj.keys():
            if len(dj[key]) > threshold:
                fj += 1
        if plot_flag:
            plt.show()
        return fj / len(list(dj.keys()))


def clip1(x):
    return max(0, min(1, x))


def clip10(x):
    return max(0, min(10, x))


# prints all the calculated metrics
class Scoring:
    def __init__(self, fs: FormScore, pol="rep_and_threshold"):
        self.fs = fs
        self.pol = Policy
        a, b, c = fs.peak_and_tempo()
        if pol == "rep_and_threshold":
            # print(self.fs.sudden_release())
            sr_metric = Policy.rep_and_threshold(self.fs.y, fs.sudden_release(),
                                                 0)  # min(10, (self.sudden_release()) / (2 * len(a)))
            jitter_metric = Policy.rep_and_threshold(self.fs.y, self.fs.jitter(), 4)

        blip_jitter_metric = fs.smooth_blips()

        it_metric = math.sqrt(c) / b
        if print_flag:
            print("\nMETRICS\nSudden Release metric: ", 10 * (sr_metric), ",\nJitter Metric: ", 10 * (jitter_metric),
                  ",\nInconsistent Tempo Metric: ", 10 * (it_metric),
                  ",\nBlips Jitter Metric: ", 10 * (blip_jitter_metric)
                  )

        blip_jitter_metric = 1 - fs.smooth_blips()

        d = {}
        d["power"] = fs.power()
        d["form"] = {}
        d["form"]["sudden_metric"] = 10 * clip1(sr_metric)
        d["form"]["jitter_metric"] = 10 * clip1(jitter_metric)
        d["form"]["it_metric"] = 10 * clip1(it_metric)

        d["ring stamina"] = clip10(fs.rings_stamina_v2())
        # d["ipf"] = fs.ipf_gl_coeff(bwt=fs.h_params["body_weight"],gender=fs.h_params["gender"],mode=fs.h_params["mode"])

        d["form"]["blip_jitter_metric"] = 10 * clip1(blip_jitter_metric)

        d["global_score"] = fs.h_params["global_score"]
        d["rep_score"] = 33.333 * (
                    fs.h_params["l0"] * sr_metric + 0 * jitter_metric + fs.h_params["l2"] * it_metric + fs.h_params[
                "l3"] * blip_jitter_metric)
        d["global_score"] = (1 - fs.h_params["discount"]) * d["rep_score"] + d["global_score"] * (
        fs.h_params["discount"])
        d["ring stamina"] = fs.rings_stamina_v2()

        coaching_tip = ""
        if it_metric > 0.5:
            coaching_tip += "Make sure each set is of 4 seconds!"
        else:
            coaching_tip += "Good rhythm! Keep it up!"

        if jitter_metric > 0.5:
            coaching_tip += " Do not shake. Hold the equipment firmly."
        else:
            coaching_tip += " Good grip! Keep it up!"

        if sr_metric > 0.3:
            coaching_tip += " Do not release the equipment in a single session"
        else:
            coaching_tip += " Good job holding the equipment! Keep it up!"

        d["coaching_tip"] = coaching_tip

        self.d = d
        import time
        t = time.time()
        d["timeID"] = str(t)


    def scores(self):
        return self.d


# In[ ]:




