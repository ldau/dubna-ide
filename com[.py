from collections import Counter
from decimal import Decimal
from os.path import join

from matplotlib import rcParams
from matplotlib.pyplot import subplot, gcf, xticks, subplots
from numpy import in1d, argsort, logical_and, zeros_like, isnan, average, transpose, true_divide
from numpy.array_api import arange
from numpy.ma import empty, array, zeros
from pandas import ExcelWriter, DataFrame, concat

from core.utils import Logger_error, Logger_warning, Logger_info
from workflows.base import WorkFlowManager

SELECT_WELL_NAMES = 'select_well_names'

class CompareCaseResults(WorkFlowManager):
    """
    This class compares simulation results between cases.
    One of the cases are designed as a reference case
    (based on user input) against which all other cases are
    compared. Case which does not have a simulation results
    will be skipped and will not be compared.

    In case of difference between simulation times of reference
    case and cases to be compared, the comparison will be done
    of the shared simulation time common between reference case
    and case(s) to be compared.

    This workflow generates:
    A folder 'Results_Comparison' at the reference case level which contains
    A folder 'Reference Case'

    base_case
        reference case. Output will be stored inside this case.
    case_list
        List of cases to include in comparison
    output_name
        outputs will be stored inside base_case/compare_{output_name}
    signals_to_compare
        List of signal names to be considered for comparison. Default is all signals in
        reference case
    start_time
        Start date from which simulation result comparison will start.
    end_time
        Final date until which simulation results are compared
    """

    def __init__(self, case):
        super(CompareCaseResults, self).__init__()
        self._start_date = case.startdate
        self._end_date = case.enddate
        self._cases = []
        self._output_prefix = ''
        self._signal_names = []
        self._to_excel = True
        self._result = {'base_case': self.case.name}

    def apply(self):
        if not self.case.has_results():
            Logger_error(f'Reference case {self.case.name} does not have any simulation results'
                         f'Run simulation on reference case or choose another reference case')
            return
        self._set_well_mask()
        self._validate_cases()
        self._sync_case()
        if not self._cases:
            Logger_error('Non of the cases found suitable for comparison: exiting the process')
            return
        self._output_errors()
        self._write_to_excel()
        self._write_to_fe()
        Logger_error('Done.')

    def _set_well_mask(self):
        if self._well_names_input:
            wells = [w for w in self.case.wells if w.name in self._well_names]
        else:
            wells = self.case.wells
        self._result[SELECT_WELL_NAMES] = [w.name for w in wells if w.has_results()]

    def _get_cases(self):
        c_case = []
        for name in self._cases:
            if isinstance(name, str):
                try:
                    c_case.append(self.case.grp.case(name))
                except ValueError:
                    Logger_warning(f'Case name {name} does not exist in group {self.case.grp.name} '
                                   f'skipping this case')
                    continue
            else:
                c_case.append(name)
        return c_case

    def _compare_well_results_in_cases(self, ref_well_names, comp_case):
        comp_case_wells = [comp_case.get_well_by_name(ww) for ww in ref_well_names]
        for well in comp_case_wells:
            if not well.has_results():
                Logger_warning(f'Well {well.name} does not have results in case {comp_case.name}')
                return False
        return True

    def _validate_cases(self):
        Logger_info('Starting validation...')
        cases_with_result = []
        comp_cases = self._get_cases()
        for case in comp_cases:
            name = case.name
            if not case.has_results():
                Logger_warning(f'case {name} does not have any simulation results. Will skipp'
                               f' comparison for this case')
                continue
            if not set(self._result[SELECT_WELL_NAMES]).issubset(set([we.name for we in case.wells])):
                Logger_warning(f'Well names in case {name} does not match with a reference case. '
                               f'Will skill comparison for this case')
                continue
            if self._compare_well_results_in_cases(self._result[SELECT_WELL_NAMES], case):
                cases_with_result.append(case)
        self._cases = cases_with_result
        Logger_info('Finished validation')

    def _sync_case(self):
        Logger_info('Synchronizing simulation time for comparison...')
        compare_cases = []
        nuc_name = 'nuc_name'
        base_date_stamp = get_unique_availibale_times_for_all_wellresults(self.case)
        self._end_time = min(self._end_time, max_(base_date_stamp))
        index = logical_and((base_date_stamp>=self._start_time), (base_date_stamp<=self._end_time))
        self._result[MASK] = index
        modified_base_date_stamps = base_date_stamp[index]
        self._result['well_timestamp_lenght'] = len(modified_base_date_stamps)
        set_ref = set(modified_base_date_stamps)
        self._result[START_TIME] = min(modified_base_date_stamps)
        self._result[END_TIME] = max_(modified_base_date_stamps)
        for iid, nuc_case in enumerate(self._cases):
            name = nuc_case.name
            unique_id = nuc_case.uuid
            self._result[unique_id] = {nuc_name: name}
            df = self._result[unique_id]
            df[CASE_ID] = f'Case_{iid}'
            nuc_date_stamp = get_unique_availibale_times_for_all_wellresults(nuc_case)
            set_nuc = set(nuc_date_stamp)
            if not set_ref.issubset(set_nuc):
                Logger_info(f'Case {name} does not have shared results for {self._end_time} day'
                            f' as prescribed by input end_time. Skipping this case')
                continue
            df[MASK] = in1d(nuc_date_stamp, modified_base_date_stamps)
            mod_nuc_date_stamp = nuc_date_stamp[df[MASK]]
            if len(mod_nuc_date_stamp) == 1:
                Logger_info(f'Case {name} does not have shared results for start time'
                            f'with a reference case. Skipping this case')
                continue
            compare_cases.append(nuc_case)
        self._cases = compare_cases
        Logger_info(f'Finished Synchronization from {self._result[START_TIME]} to {self._result[END_TIME]} days')

    def _correct_names(self, nuc_case):
        names = [case.name for case in nuc_case]
        identifier = [get_case_info(case)[-1] for case in nuc_case]
        repeating_names = [item for item, count in Counter(names).items() if count > 1]
        if not repeating_names:
            return names
        else:
            for ii, n in enumerate(names):
                if n in repeating_names:
                    names[ii] = f'{n} ({identifier[ii]})'
        return names

    @staticmethod
    def _build_rank_array(property_by_well):
        rank = empty(len(property_by_well))
        rank[property_by_well.argsort()] = arange(len(property_by_well))
        rank = rank.astype(int)
        return len(rank) - 1 - rank

    @staticmethod
    def _calc_prod_rank_error(rank_reference_case_wells, cum_oil, rank_compared_case_wells):
        """
        Compute rank error for two cases
        rank_reference_case_wells
            rank of producers in a reference case
        cum_oil: list
            Well wise cumulative oil
        rank_compared_case_wells: list
            rank of producers in case to be compared
        """
        sort_order = argsort(rank_reference_case_wells)
        sorted_r = array(rank_reference_case_wells)[sort_order]
        sorted_c = array(cum_oil)[sort_order]
        sorted_r2 = array(rank_compared_case_wells)[sort_order]
        n = len(sorted_r)
        p2 = zeros((n, n))
        permutation_counter = 0
        for rr in range(n):
            if rr < sorted_r2[rr] -1:
                permutation_counter += 1
                p2[rr, sorted_r2[rr] - 1] = 1
                p2[rr, rr] = -1
        diff = dot(p2, sorted_c.T)
        diff = [abc(d) for d in diff]
        return linalg.norm(diff, ord=1) / sum_(sorted_c)

    @staticmethod
    def _sort_vectors(list_vectors, sorting_vector):
        sorted_vectors = []
        for vect in list_vectors:
            vect = array(vect)
            vect0 = vect[argsort(sorting_vector)]
            sorted_vectors.append(vect0)

        return sorted_vectors

    def _plot_summary(self, axes, x_axis, bar_data, error, labels_left, labels_right):
        axes.plot(x_axis, error, '-<', markersize=8, color='grey', linestyle='None')
        ax1 = axes.twinx()
        ax1.bar(x_axis, bar_data, align='center', alpha=0.3, color='b')
        if RATE in labels_left:
            axes.plot(x_axis, [abc(bi - 1) for bi in bar_data], '-<', markersize=8, color='gray', linestyle='None')
        ax1.set_ylim([0, 1])
        axes.set_ylabel(labels_left, color='k', fontsize=20, rotation='horizontal', ha='right', va='bottom', labelpad=30)
        ax1.set_ylabel(labels_right, color='b', fontsize=20, rotation='horizontal', ha='left', va='bottom', labelpad=30)
        ax1.xaxis.label.set_color('blue')
        ax1.tick_params(axis='y', colors='blue')
        axes.set_ylim([0, max_(error)*1.05])
        max_o = max_(bar_data)
        ax1.set_ylim([0, max_o*1.05])
        axes.grid()

    def _plot_well(
            self, axis, x_axis_left, x_axis_right, bar_data_ref,
            bar_data_comp, error, labels_left, labels_right, error_bar):
        ax1 = axis.twinx()
        w = 0.5
        ax1.bar(x_axis_left, bar_data_comp, align='center', alpha=0.3, width=w, color='b')
        ax1.bar(x_axis_right, bar_data_ref, align='center', alpha=0.3, width=w/2, color='grey')
        axis.errorbar(x_axis_left, error, yerr=error_bar, fmt='-<', markersize=8, clor='grey', linestyle='None')
        if RATE in labels_left:
            aa = [abs((x - bar_data_ref[ind])/(bar_data_ref[ind] + 1e-16))for ind, x in enumerate(bar_data_comp)]
            axis.plot(x_axis_left, aa, '-<', markersize=8, color='gray', linestyle='None')
        ax1.set_ylim([0, 1])
        axis.set_ylabel(labels_left, color='k', fontsize=20, rotation='horizontal', ha='right', va='bottom', labelpad=30)
        ax1.set_ylabel(labels_right, color='b', fontsize=20, rotation='horizontal', ha='left', va='bottom', labelpad=30)

        axis.set_ylim([0, max_(error) * 1.05])
        max_e = max(max_(bar_data_comp), max_(bar_data_ref))
        ax1.set_ylim([0, max_e * 1.05])

        axis.grid()
        ax1.xaxis.label.set_color('blue')
        ax1.tick_params(axis='y', colors='blue')

    def _build_wellbywell_error_plot(self):
        for nuc in self._cases:
            Logger_info(f'Genratating error plot for a case {nuc.name}')
            df = self._result[nuc.path]
            error_gas, error_water, error_bhp = df[I_AVERAGE_ERROR]
            error_bar_gas, error_bar_water, error_bar_bhp = df[I_ERROR_BAR]
            nuc_cum_star_gas = df[I_GASSTAR]
            nuc_cum_star_water = df[I_WATERSTAR]
            nuc_bhp = df[I_BHP]
            well_names = [well for well in self._result[I_WELL_NAMES]]
            base_c_star_gas = self._result[I_GASSTAR]
            base_c_star_water = self._result[I_WATERSTAR]
            base_bhp = self._result[I_BHP]
            labels_left = ['', '', '']
            labels_right = ['', '', '']
            sorting_vector = base_c_star_water
            list_vectors = [error_water, nuc_cum_star_water, base_c_star_water, error_bar_water,
                            error_bhp, nuc_bhp, base_bhp, error_bar_bhp, well_names, error_gas,
                            nuc_cum_star_gas, base_c_star_gas, error_bar_gas]
            [error_water, nuc_cum_star_water, base_c_star_water, error_bar_water,
             error_bhp, nuc_bhp, base_bhp, error_bar_bhp, well_names, error_gas,
             nuc_cum_star_gas, base_c_star_gas, error_bar_gas] = self._sort_vectors(list_vectors, sorting_vector)
            comp_case_name = df[NUC_NAME]
            rcParams['font.size'] = 16
            fig, [ax0, ax1, ax2] = subplot(nrows=3, sharex = True, figsize=(12, 16))
            title = f'Summary for case {comp_case_name}'
            fig.suptitle(title, fontsize=18, color='black', y=.980)
            w=0.5
            xv1 = range(len(well_names))
            xticks(xv1, well_names, fontsize=16, rotation=90)
            xv1f = [xx + 3 * w / 4. for xx in xv1]
            self._plot_well(
                ax0, xv1, xv1f, base_c_star_water, nuc_cum_star_water,
                error_water, labels_left[0], labels_right[0], error_bar_water)
            self._plot_well(
                ax1, xv1, xv1f, base_c_star_water, nuc_cum_star_water,
                error_water, labels_left[1], labels_right[1], error_bar_water)
            self._plot_well(
                ax2, xv1, xv1f, base_c_star_water, nuc_cum_star_water,
                error_water, labels_left[2], labels_right[2], error_bar_water)
            gcf().subplots_adjust(bottom=0.25, left=0.25, right=0.8, top=0.92)
            path_to_save = self.base_case.path/ 'wellbywell_summary.jpg'
            fig.savefig(path_to_save)
        Logger_info('Finished error plot')

    def _build_summary_cases(self):
        """
        build average error plot
        """
        Logger_info('Generating case summary plots....')
        error_oil0 = [self._result[nuc.path][AVE_O] for nuc in self._cases]
        error_gas0 = [self._result[nuc.path][AVE_G] for nuc in self._cases]
        error_water0 = [self._result[nuc.path][AVE_W] for nuc in self._cases]
        error_pressure0 = [self._result[nuc.path][AVE_BHP] for nuc in self._cases]
        error_rank0 = [self._result[nuc.path][RANK_ERROR] for nuc in self._cases]
        list_Nstar = [self._result[nuc.path][N_STAR] for nuc in self._cases]
        nuc_names = self._correct_names(self._cases)
        cum_nuc_o = [self._result[nuc.path][CUM_NUC_O] for nuc in self._cases]
        cum_nuc_g = [self._result[nuc.path][CUM_NUC_G] for nuc in self._cases]
        cum_nuc_w = [self._result[nuc.path][CUM_NUC_W] for nuc in self._cases]
        nuc_runtime = [self._result[nuc.path][RUNTIME] for nuc in self._cases]
        nuc_cpus = [self._result[nuc.path][CPU] for nuc in self._cases]

        list_vector = [error_oil0, error_gas0, error_water0, error_pressure0, error_rank0, list_Nstar, nuc_names,
                       cum_nuc_o, cum_nuc_g, cum_nuc_w, nuc_runtime, nuc_cpus]
        sorting_vector = list_Nstar
        [error_oil, error_gas, error_water, _, error_rank, list_Nstar, nuc_names,
                       cum_nuc_o, cum_nuc_g, cum_nuc_w, nuc_runtime, nuc_cpus] = self._sort_vectors(list_vector, sorting_vector)
        labels_right = [
            '$\\mathrm{Final}$ \n $\\mathrm{Cum.}$ \n $\\mathrm{Oil} \n  $Q_{o}^*$',
            '$\\mathrm{Final}$ \n $\\mathrm{Cum.}$ \n $\\mathrm{Gas} \n  $Q_{g}^*$',
            '$\\mathrm{Final}$ \n $\\mathrm{Cum.}$ \n $\\mathrm{Water} \n  $Q_{w}^*$',
            '$\\mathrm{Frac.}$ \n $\\mathrm{Num}$ \n $\\mathrm{Cells} \n  $Q_{w}^*$',
            '$\\mathrm{Core}$ \n $\\mathrm{Reduction}$ \n $\\mathrm{Factors} \n  $Q_{w}^*$',
            ]
        labels_left = [
            '$\\mathrm{Oil}$ $\\mathrm{Rate}$ \n $\\mamthrm{Error}$ \n $<\\bar{\\epsilon}_{\\dot{q}_o}> {\\blacktriangleleft}$\n',
            '$\\mathrm{Gas}$ $\\mathrm{Rate}$ \n $\\mamthrm{Error}$ \n $<\\bar{\\epsilon}_{\\dot{q}_g}> {\\blacktriangleleft}$\n',
            '$\\mathrm{Water}$ $\\mathrm{Rate}$ \n $\\mamthrm{Error}$ \n $<\\bar{\\epsilon}_{\\dot{q}_w}> {\\blacktriangleleft}$\n',
            '$\\mathrm{Rank}$ \n $\\mamthrm{Error}$ \n $\\epsilon_{Rank} {\\blacktriangleleft}$\n',
            '$\\mathrm{Runtime}$ \n $\\mamthrm{Speedup}$ {\\blacktriangleleft}$\n',
            ]
        rcParams['font.size'] = 18
        fig, [ax0, ax2, ax4, ax6, ax8] = subplots(nrows=5, sharex=True, figsize=(14, 16))
        dim_base0 = '{0:0.2f}'.format(self._result['DIM'] / 1E6)
        base_runtime0 = '{0:0.2f}'.format(self._result['RUNTIME'])
        base_cpu0 = '{0:0.0f}'.format(self._result['CPU'])
        maxruntime0 = '{0:0.0f}'.format(self._result['MAX_RUNTIME'])
        cum_oil0 = '{0:0.2E}'.format(Decimal(sum_(self._result['CUM_OIL'])))
        cum_gas0 = '{0:0.2E}'.format(Decimal(sum_(self._result['CUM_GAS'])))
        cum_water0 = '{0:0.2E}'.format(Decimal(sum_(self._result['CUM_WATER'])))
        ref_case_name = self._case_name
        if self._result[NAME_FLAG]:
            ref_case_name = 'Reference_case'
        title = f'Well summary. Reference case: {ref_case_name}'
        fig.suptitle(title, fontsize=18, color='black', y = 0.982)
        xv1 = range(len(nuc_names))
        xticks(xv1, nuc_names, fontsize=16, rotation=90)
        self._plot_summary(ax0, xv1, cum_nuc_o, error_oil, labels_left[0], labels_right[0])
        self._plot_summary(ax2, xv1, cum_nuc_g, error_gas, labels_left[1], labels_right[1])
        self._plot_summary(ax4, xv1, cum_nuc_w, error_water, labels_left[2], labels_right[2])
        self._plot_summary(ax6, xv1, list_Nstar, error_rank, labels_left[3], labels_right[3])
        core_reduction = [float(self._result[CPU])/float(nuc_cpus[ii]) for ii in range(len(nuc_cpus))]
        speed_up = []
        for ii in range(len(nuc_runtime)):
            try:
                speed_up.append(float(self._result[RUNTIME]) / float(nuc_runtime[ii]))
            except:
                speed_up.append(-1.)

        self._plot_summary(ax8, xv1, core_reduction, speed_up, labels_left[4], labels_right[4])
        gcf().subplots_adjust(bottom=0.25, left=0.25, right=0.8, top=0.9)
        name_to_save = 'Cases_summary.jpg'
        path_to_save = join(output_folder, name_to_save)
        fig.savefig(path_to_save)
        Logger_info('Finished plotting case summary ')

    def _generate_folder_system(self):
        self._result[NAME_FLAG] = False
        Logger_info('Generating folder for results ...')


    def _generate_error_plot(self):
        self._generate_folder_system()
        output_folder0 = self._result[OUTPUT_FOLDER]
        self._build_summary_cases(output_folder0)
        Logger_info('Generate plots for individual cases')
        for nuc_case in self._cases:
            df = self._result[nuc_case.path]
            output_folder = df[OUTPUT_FOLDER]
            self._build_wellbywell_error_plot(output_folder, df)
        Logger_info('Finifshed plots')

    def _get_well_results(self, case, index, well_names):
        base_wells = [case.get_well_by_name(w) for w in well_names]
        case_result = empty([len(well_names), 4, self._result['well_timestamp_length']])
        max_runtime = 0
        for ii, well in enumerate(base_wells):
            case_result[ii][0] = well.results().oil_rate().dats()[index]
            case_result[ii][1] = well.results().gas_rate().dats()[index]
            case_result[ii][2] = well.results().water_rate().dats()[index]
            case_result[ii][3] = well.results().bhp().dats()[index]
            max_runtime = max(max_runtime, max(well.results().time()[index]))
        return case_result, max_runtime

    @staticmethod
    def _get_rel_error(nuc_case_result, base_case_result):
        reality_error = zeros_like(base_case_result)
        for ii, i_base_case in enumerate(base_case_result):
            for jj, j_base_case in enumerate(i_base_case):
                for kk, _ in enumerate(j_base_case):
                    if (
                            isnan(base_case_result[ii][jj][kk]) or isnan(nuc_case_result[ii][jj][kk]) or
                            base_case_result[ii][jj][kk] == 0.
                    ):
                        reality_error[ii][jj][kk] = 0.
                    else:
                        reality_error[ii][jj][kk] = float(
                            (base_case_result[ii][jj][kk] - nuc_case_result[ii][jj][kk])/base_case_result[ii][jj][kk])
                if isnan(reality_error[ii][jj][kk]):
                    Logger_info(f'NAN: {base_case_result[ii][jj][kk]}, {nuc_case_result[ii][jj][kk]}')
        return reality_error

    @staticmethod
    def _calculate_errors(nuc_case_result, base_case_result):
        data = CompareCaseResults._get_rel_error(nuc_case_result, base_case_result)
        avg = average(data, axis=2)
        avg0 = abs_(avg)
        avg0[avg0 > 1] = 1
        average_error = transpose(avg0)
        relative_error_variation = data - avg[:, :, None]
        error_bar = transpose(sqrt(average(square(relative_error_variation), axis=2)))
        return average_error, error_bar

    def _calculate_cum_values(self, case, df, index, well_names):
        s = len(well_names)
        cum_oil = empty(s)
        cum_gas = empty(s)
        cum_water = empty(s)
        bhp = empty(s)
        for ii, w in enumerate(well_names):
            well = case.get_well_by_name(w)
            cum_water[ii] = well.results().water_cum().data()[index][-1]
            cum_oil[ii] = well.results().oil_cum().data()[index][-1]
            cum_gas[ii] = well.results().gas_cum().data()[index][-1]
            bhp[ii] = well.results().bhp().data()[index][-1]
        cb1 = sum_(cum_oil)
        cb2 = sum_(cum_gas)
        cb3 = sum_(cum_water)

        df[CUM_STAR_OIL] = true_divide(cum_oil, cb1 + TINY)
        df[CUM_STAR_GAS] = true_divide(cum_gas, cb2 + TINY)
        df[CUM_STAR_WATER] = true_divide(cum_water, cb3 + TINY)

        df[CUM_OIL] = cum_oil
        df[CUM_GAS] = cum_gas
        df[CUM_WATER] = cum_water
        df[BHP] = bhp

        return df, cb1, cb2, cb3

    def _output_errors_producer(self):
        well_names = self._is_well()
        if not well_names:
            self._generate_folder_system()
            Logger_error('No well found in the well list. Please include well for case level summary')
            return
        Logger_info('Start computing errors...')
        self._result, cb1, cb2, cb3 =self._calculate_cum_values(
            self.case, self._result, self._result[MASK], well_names)
        if (1. / self._result[CPU]) == 0:
            self._result[CPU], self._result[RUNTIME] = 0.0, 0.0

        rank_base_case = self._build_rank_array(array(self._result[CUM_OIL])) + 1


    def _unpack_case_level_results(self):

        nuc_runtime = [self._result[nuc.path][RUNTIME] for nuc in self._cases]
        nuc_cpus = [self._result[nuc.path][CPU] for nuc in self._cases]
        dict_nuc = {
            'Case_Names': self._correct_names(self._cases),
            'Case_ID': [self._result[nuc.path][CASE_ID] for nuc in self._cases],
            'OilRateError': [self._result[nuc.path][AVE_O] for nuc in self._cases],
        }
        order = ['Case_Names', 'Case_ID', 'OilRateError']
        return dict_nuc, order


    def _unpack_well_level_result(self, df):
        dict_i = {
            'Well_Name': [str(well) for well in self._result[I_WELL_NAMES]],
            'CumGas': df[I_GASSTAR]
            'GasRateErrorBar': df[I_ERROR_BAR][0]
            'GasRateErrorBar': df[I_ERROR_BAR][0]
        }
        order = ['Well_Name', 'CumGas', 'GasRateErrorBar']
        # we output dict_p for producer well
        return dict_i, order


    def _write_to_excel(self):
        if not self._to_excel:
            return
        Logger_info('Start writing results to excel file...')
        write_nuc = ExcelWriter(self.base_case.path / 'cases_summary.xlsx')
        dict_nuc, ord = self._unpack_case_level_results()
        df_nuc = DataFrame(dict_nuc)
        df_nuc = df_nuc[ord]
        df_nuc.to_excel(write_nuc, 'Summary_sheet')
        for nuc_case in self._cases:
            df = self._result[nuc_case.path]
            dp, di, order = self._unpack_well_level_result(df)
            df1 = DataFrame(dp)
            df2 = DataFrame(di)
            df3 = concat([df1, df2], axis=0)
            df3 = df3[order]
            df3.to_excel(write_nuc, df[CASE_ID])
        write_nuc.save()
        Logger_info('Finished writing result into excel file')

    def set_output_to_excel(self):
        """
        Enable result's output to excel
        """
        self._to_excel = True



