import sys

import numpy as np

import openmdao.api as om

import pycycle.api as pyc

from PW1133_Fan_map import FanMap
from PW1133_LPC_map import LPCMap
from PW1133_HPC_map import HPCMap
from PW1133_HPT_map import HPTMap
from PW1133_LPT_map import LPTMap

class PW1133(pyc.Cycle):

    def initialize(self):
        self.options.declare('cooling', default=False,
                              desc='If True, calculate cooling flow values.')

        super().initialize()

    def setup(self):

        USE_TABULAR = False   #True run the tabular version (gas table equivalent)

        if USE_TABULAR: 
            self.options['thermo_method'] = 'TABULAR'
            self.options['thermo_data'] = pyc.AIR_JETA_TAB_SPEC
            FUEL_TYPE = "FAR"
        else: 
            self.options['thermo_method'] = 'CEA'
            self.options['thermo_data'] = pyc.species_data.janaf
            FUEL_TYPE = "Jet-A(g)"
        
        cooling = self.options['cooling']
        design = self.options['design']


        # Adding Subsystems

        # Core Sections
        self.add_subsystem('fc', pyc.FlightConditions())
        self.add_subsystem('inlet', pyc.Inlet())
        self.add_subsystem('fan', pyc.Compressor(map_data=FanMap, map_extrap=True,
                                                 bleed_names=[]),
                           promotes_inputs=[('Nmech','Fan_Nmech')])
        self.add_subsystem('splitter', pyc.Splitter())
        self.add_subsystem('duct4', pyc.Duct(expMN=2.0, ))
        self.add_subsystem('lpc', pyc.Compressor(map_data=LPCMap, map_extrap=True),
                            promotes_inputs=[('Nmech','LP_Nmech')])
        self.add_subsystem('duct6', pyc.Duct(expMN=2.0, ))
        self.add_subsystem('hpc', pyc.Compressor(map_data=HPCMap, map_extrap=True,
                                        bleed_names=['cust','bld_inlet','bld_exit']),
                           promotes_inputs=[('Nmech','HP_Nmech')])
        self.add_subsystem('bld3', pyc.BleedOut(bleed_names=['bld_inlet','bld_exit']))
        self.add_subsystem('burner', pyc.Combustor(fuel_type=FUEL_TYPE))
        hpt = self.add_subsystem('hpt', pyc.Turbine(map_data=HPTMap, map_extrap=True,
                                              bleed_names=['bld_inlet','bld_exit']),
                           promotes_inputs=[('Nmech','HP_Nmech')])
        hpt.set_input_defaults('PR', val=3.964)
        self.add_subsystem('duct11', pyc.Duct(expMN=2.0, ))
        lpt = self.add_subsystem('lpt', pyc.Turbine(map_data=LPTMap, map_extrap=True,
                                              bleed_names=['bld_inlet','bld_exit']),                                    
                           promotes_inputs=[('Nmech','LP_Nmech')])
        lpt.set_input_defaults('PR', val=8.981) 
        self.add_subsystem('duct13', pyc.Duct(expMN=2.0, ))
        self.add_subsystem('core_nozz', pyc.Nozzle(nozzType='CV', lossCoef='Cv', ))

        # Bypass Sections
        self.add_subsystem('byp_bld', pyc.BleedOut(bleed_names=['bypBld']))
        self.add_subsystem('duct15', pyc.Duct(expMN=2.0, ))
        self.add_subsystem('byp_nozz', pyc.Nozzle(nozzType='CV', lossCoef='Cv', ))

        # Shafts
        self.add_subsystem('fan_shaft', pyc.Shaft(num_ports=2), promotes_inputs=[('Nmech','Fan_Nmech')])
        self.add_subsystem('gearbox', pyc.Gearbox(), promotes_inputs=[('N_in','LP_Nmech'), ('N_out','Fan_Nmech')])
        self.add_subsystem('lp_shaft', pyc.Shaft(num_ports=3), promotes_inputs=[('Nmech','LP_Nmech')])
        self.add_subsystem('hp_shaft', pyc.Shaft(num_ports=2), promotes_inputs=[('Nmech','HP_Nmech')])
        self.add_subsystem('perf', pyc.Performance(num_nozzles=2, num_burners=1))

        # Performance characteristics
        self.connect('inlet.Fl_O:tot:P', 'perf.Pt2')
        self.connect('hpc.Fl_O:tot:P', 'perf.Pt3')
        self.connect('burner.Wfuel', 'perf.Wfuel_0')
        self.connect('inlet.F_ram', 'perf.ram_drag')
        self.connect('core_nozz.Fg', 'perf.Fg_0')
        self.connect('byp_nozz.Fg', 'perf.Fg_1')

        # Fan-shaft connections
        self.connect('fan.trq', 'fan_shaft.trq_0')
        self.connect('gearbox.trq_out', 'fan_shaft.trq_1')

        # LP-shaft connections
        self.connect('gearbox.trq_in', 'lp_shaft.trq_0')
        self.connect('lpc.trq', 'lp_shaft.trq_1')
        self.connect('lpt.trq', 'lp_shaft.trq_2')

        # HP-shaft connections
        self.connect('hpc.trq', 'hp_shaft.trq_0')
        self.connect('hpt.trq', 'hp_shaft.trq_1')

        #Ideally expanding flow by conneting flight condition static pressure to nozzle exhaust pressure
        self.connect('fc.Fl_O:stat:P', 'core_nozz.Ps_exhaust')
        self.connect('fc.Fl_O:stat:P', 'byp_nozz.Ps_exhaust')

        # Needed?
        self.add_subsystem('ext_ratio', om.ExecComp('ER = core_V_ideal * core_Cv / ( byp_V_ideal *  byp_Cv )',
                        core_V_ideal={'val':1000.0, 'units':'ft/s'},
                        core_Cv={'val':0.9905, 'units':None},
                        byp_V_ideal={'val':1000.0, 'units':'ft/s'},
                        byp_Cv={'val':0.9948, 'units':None},
                        ER={'val':0.9957, 'units':None}))

        self.connect('core_nozz.ideal_flow.V', 'ext_ratio.core_V_ideal')
        self.connect('byp_nozz.ideal_flow.V', 'ext_ratio.byp_V_ideal')


        main_order = ['fc', 'inlet', 'fan', 'splitter', 'duct4', 'lpc', 'duct6', 'hpc', 'bld3', 'burner', 'hpt', 'duct11',
                            'lpt', 'duct13', 'core_nozz', 'byp_bld', 'duct15', 'byp_nozz', 'gearbox', 'fan_shaft', 'lp_shaft', 'hp_shaft',
                            'perf', 'ext_ratio']

        balance = self.add_subsystem('balance', om.BalanceComp())

        # On Design Balances
        if design: 

            #balance.add_balance('FAR', eq_units='degR', lower=1e-4, val=0.02672)
            #self.connect('balance.FAR', 'burner.Fl_I:FAR')
            #self.connect('burner.Fl_O:tot:T', 'balance.lhs:FAR')

            # Finds Low Pressure Turbine Pressure Ratio through a net power of zero (pwr_net) on the low pressure shaft (lp_shaft)
            #balance.add_balance('lpt_PR', val=8.981, lower=1.001, upper=20, eq_units='hp', rhs_val=0., res_ref=1e4)
            #self.connect('balance.lpt_PR', 'lpt.PR')
            #self.connect('lp_shaft.pwr_net', 'balance.lhs:lpt_PR')

            # Finds High Pressure Turbine Pressure Ratio through a net power of zero (pwr_net) on the high pressure shaft (hp_shaft)
            #balance.add_balance('hpt_PR', val=3.964, lower=1.001, upper=8, eq_units='hp', rhs_val=0., res_ref=1e4)
            #self.connect('balance.hpt_PR', 'hpt.PR')
            #self.connect('hp_shaft.pwr_net', 'balance.lhs:hpt_PR')

            # Finds the base torque in the gearbox (gearbox.trq_base) that results in a net power of zero (pwr_net) on the fan shaft (fan_shaft)
            balance.add_balance('gb_trq', val=23928.0, units='ft*lbf', eq_units='hp', rhs_val=0., res_ref=1e4)
            self.connect('balance.gb_trq', 'gearbox.trq_base')
            self.connect('fan_shaft.pwr_net', 'balance.lhs:gb_trq')

            # Finds the pressure ratio across the high pressure compressor (hpc.PR) that results in a specific Overall Pressure Ratio (OPR_simple) in the engine. 
            balance.add_balance('hpc_PR', val=13.5, units=None, eq_units=None)
            self.connect('balance.hpc_PR', ['hpc.PR', 'opr_calc.HPCPR'])
            # self.connect('perf.OPR', 'balance.lhs:hpc_PR')
            self.connect('opr_calc.OPR_simple', 'balance.lhs:hpc_PR')

            # Finds the fan efficiency (fan.eff) that results in a certain polytropic efficiency  (fan.eff_poly)
            balance.add_balance('fan_eff', val=0.9286 , units=None, eq_units=None)
            self.connect('balance.fan_eff', 'fan.eff')
            self.connect('fan.eff_poly', 'balance.lhs:fan_eff')

            # Finds the low pressure compressor efficiency (lpc.eff) that results in a certain polytropic efficiency  (lpc.eff_poly)
            balance.add_balance('lpc_eff', val=0.9150, units=None, eq_units=None)
            self.connect('balance.lpc_eff', 'lpc.eff')
            self.connect('lpc.eff_poly', 'balance.lhs:lpc_eff')

            #balance.add_balance('hpc_eff', val=0.9150, units=None, eq_units=None)
            #self.connect('balance.hpc_eff', 'hpc.eff')
            #self.connect('hpc.eff_poly', 'balance.lhs:hpc_eff')

            # Finds the high pressure turbine efficiency (hpt.eff) that results in a certain polytropic efficiency (hpt.eff_poly)
            balance.add_balance('hpt_eff', val=0.8734, units=None, eq_units=None)
            self.connect('balance.hpt_eff', 'hpt.eff')
            self.connect('hpt.eff_poly', 'balance.lhs:hpt_eff')

            # Finds the low pressure turbine efficiency (lpt.eff) that results in a certain polytropic efficiency (lpt.eff_poly)
            balance.add_balance('lpt_eff', val=0.8854 , units=None, eq_units=None)
            self.connect('balance.lpt_eff', 'lpt.eff')
            self.connect('lpt.eff_poly', 'balance.lhs:lpt_eff')

            # Needed?
            self.add_subsystem('fan_dia', om.ExecComp('FanDia = 2.0*(area/(pi*(1.0-hub_tip**2.0)))**0.5',
                            area={'val':4689, 'units':'inch**2'},
                            hub_tip={'val':0.3, 'units':None},
                            FanDia={'val':81, 'units':'inch'}))
            self.connect('inlet.Fl_O:stat:area', 'fan_dia.area')

            self.add_subsystem('opr_calc', om.ExecComp('OPR_simple = FPR*LPCPR*HPCPR',
                            FPR={'val':1.52, 'units':None},
                            LPCPR={'val':2.246, 'units':None},
                            HPCPR={'val':13.5, 'units':None},
                            OPR_simple={'val':46.088, 'units':None}))


            # order_add = ['hpc_CS', 'fan_dia', 'opr_calc']
            order_add = ['fan_dia', 'opr_calc']

        # Off Design Balances
        else:
            
            # Finds the fuel-air ratio (burner.Fl_I:FAR) that results in a certain net thrust (perf.Fn)
            balance.add_balance('FAR', val=0.017, lower=1e-4, eq_units='lbf')
            self.connect('balance.FAR', 'burner.Fl_I:FAR')
            self.connect('perf.Fn', 'balance.lhs:FAR')
            # self.connect('burner.Fl_O:tot:T', 'balance.lhs:FAR')

            # Finds the mass flow rate (fc.W) that results in a specific throat area (core_nozz.Throat:stat:area)
            balance.add_balance('W', units='lbm/s', lower=10., upper=2500., eq_units='inch**2')
            self.connect('balance.W', 'fc.W')
            self.connect('core_nozz.Throat:stat:area', 'balance.lhs:W')

            # Finds the bypass ratio (splitter.BPR) that results in a specific fan map line (fan.map.RlineMap)
            balance.add_balance('BPR', lower=10., upper=40.)
            self.connect('balance.BPR', 'splitter.BPR')
            self.connect('fan.map.RlineMap', 'balance.lhs:BPR')

            # Finds the rotational speed of the fan (Fan_Nmech) that results in a fan shaft net power of zero (pwr_net)
            balance.add_balance('fan_Nmech', val=2000.0, units='rpm', lower=500., eq_units='hp', rhs_val=0., res_ref=1e2)
            self.connect('balance.fan_Nmech', 'Fan_Nmech')
            self.connect('fan_shaft.pwr_net', 'balance.lhs:fan_Nmech')

            # Finds the rotational speed of the low pressure shaft (LP_Nmech) that results in a low pressure shaft net power of zero (pwr_net)
            balance.add_balance('lp_Nmech', val=6000.0, units='rpm', lower=500., eq_units='hp', rhs_val=0., res_ref=1e2)
            self.connect('balance.lp_Nmech', 'LP_Nmech')
            self.connect('lp_shaft.pwr_net', 'balance.lhs:lp_Nmech')

            # Finds the rotational speed of the high pressure shaft (HP_Nmech) that results in a high pressure shaft net power of zero (pwr_net)
            balance.add_balance('hp_Nmech', val=20000.0, units='rpm', lower=500., eq_units='hp', rhs_val=0., res_ref=1e2)
            self.connect('balance.hp_Nmech', 'HP_Nmech')
            self.connect('hp_shaft.pwr_net', 'balance.lhs:hp_Nmech')

            order_add = []

        # Cooling Balances
        if cooling:
            self.add_subsystem('hpt_cooling', pyc.TurbineCooling(n_stages=2, thermo_data=pyc.species_data.janaf, T_metal=2460.))
            self.add_subsystem('hpt_chargable', pyc.CombineCooling(n_ins=3)) #Number of cooling flows which are chargable

            self.pyc_connect_flow('bld3.bld_inlet', 'hpt_cooling.Fl_cool', connect_stat=False)
            self.pyc_connect_flow('burner.Fl_O', 'hpt_cooling.Fl_turb_I')
            self.pyc_connect_flow('hpt.Fl_O', 'hpt_cooling.Fl_turb_O')

            self.connect('hpt_cooling.row_1.W_cool', 'hpt_chargable.W_1')
            self.connect('hpt_cooling.row_2.W_cool', 'hpt_chargable.W_2')
            self.connect('hpt_cooling.row_3.W_cool', 'hpt_chargable.W_3')
            self.connect('hpt.power', 'hpt_cooling.turb_pwr')

            # Finds the fraction of the total flow that is used for non-chargable cooling (bld3.bld_inlet:frac_W) that results in a certain mass flow rate (bld3.bld_inlet:stat:W)
            #balance.add_balance('hpt_nochrg_cool_frac', val=0.03731, lower=0.02, upper=.15, eq_units='lbm/s')
            #self.connect('balance.hpt_nochrg_cool_frac', 'bld3.bld_inlet:frac_W')
            #self.connect('bld3.bld_inlet:stat:W', 'balance.lhs:hpt_nochrg_cool_frac')
            #self.connect('hpt_cooling.row_0.W_cool', 'balance.rhs:hpt_nochrg_cool_frac')

            # Finds the fraction of the total flow that is used for chargable cooling (bld3.bld_exit:frac_W) that results in a certain mass flow rate (bld3.bld_exit:stat:W)
            #balance.add_balance('hpt_chrg_cool_frac', val=0.06108, lower=0.02, upper=.15, eq_units='lbm/s')
            #self.connect('balance.hpt_chrg_cool_frac', 'bld3.bld_exit:frac_W')
            #self.connect('bld3.bld_exit:stat:W', 'balance.lhs:hpt_chrg_cool_frac')
            #self.connect('hpt_chargable.W_cool', 'balance.rhs:hpt_chrg_cool_frac')

            order_add = ['hpt_cooling', 'hpt_chargable']



        self.set_order(main_order + order_add + ['balance'])

        # Set up all the flow connections:
        self.pyc_connect_flow('fc.Fl_O', 'inlet.Fl_I')
        self.pyc_connect_flow('inlet.Fl_O', 'fan.Fl_I')
        self.pyc_connect_flow('fan.Fl_O', 'splitter.Fl_I')
        self.pyc_connect_flow('splitter.Fl_O1', 'duct4.Fl_I')
        self.pyc_connect_flow('duct4.Fl_O', 'lpc.Fl_I')
        self.pyc_connect_flow('lpc.Fl_O', 'duct6.Fl_I')
        self.pyc_connect_flow('duct6.Fl_O', 'hpc.Fl_I')
        self.pyc_connect_flow('hpc.Fl_O', 'bld3.Fl_I')
        self.pyc_connect_flow('bld3.Fl_O', 'burner.Fl_I')
        self.pyc_connect_flow('burner.Fl_O', 'hpt.Fl_I')
        self.pyc_connect_flow('hpt.Fl_O', 'duct11.Fl_I')
        self.pyc_connect_flow('duct11.Fl_O', 'lpt.Fl_I')
        self.pyc_connect_flow('lpt.Fl_O', 'duct13.Fl_I')
        self.pyc_connect_flow('duct13.Fl_O','core_nozz.Fl_I')
        self.pyc_connect_flow('splitter.Fl_O2', 'byp_bld.Fl_I')
        self.pyc_connect_flow('byp_bld.Fl_O', 'duct15.Fl_I')
        self.pyc_connect_flow('duct15.Fl_O', 'byp_nozz.Fl_I')

        # Bleed Connections 
        self.pyc_connect_flow('hpc.bld_inlet', 'lpt.bld_inlet', connect_stat=False)
        self.pyc_connect_flow('hpc.bld_exit', 'lpt.bld_exit', connect_stat=False)
        self.pyc_connect_flow('bld3.bld_inlet', 'hpt.bld_inlet', connect_stat=False)
        self.pyc_connect_flow('bld3.bld_exit', 'hpt.bld_exit', connect_stat=False)


        # Solver Set Up
        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['atol'] = 1e-4
        newton.options['rtol'] = 1e-4
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 10
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 10
        newton.options['reraise_child_analysiserror'] = False
        # newton.linesearch = om.BoundsEnforceLS()
        newton.linesearch = om.ArmijoGoldsteinLS()
        newton.linesearch.options['rho'] = 0.75
        # newton.linesearch.options['maxiter'] = 2
        newton.linesearch.options['iprint'] = -1

        self.linear_solver = om.DirectSolver()

        super().setup()

def viewer(prob, pt, file=sys.stdout):
    """
    print a report of all the relevant cycle properties
    """

    summary_data = (prob[pt+'.fc.Fl_O:stat:MN'], prob[pt+'.fc.alt'], prob[pt+'.inlet.Fl_O:stat:W'],
                    prob[pt+'.perf.Fn'], prob[pt+'.perf.Fg'], prob[pt+'.inlet.F_ram'],
                    prob[pt+'.perf.OPR'], prob[pt+'.perf.TSFC'], prob[pt+'.splitter.BPR'])

    print(file=file, flush=True)
    print(file=file, flush=True)
    print(file=file, flush=True)
    print("----------------------------------------------------------------------------", file=file, flush=True)
    print("                              POINT:", pt, file=file, flush=True)
    print("----------------------------------------------------------------------------", file=file, flush=True)
    print("                       PERFORMANCE CHARACTERISTICS", file=file, flush=True)
    print("    Mach      Alt       W      Fn      Fg    Fram     OPR     TSFC      BPR ", file=file, flush=True)
    print(" %7.5f  %7.1f %7.3f %7.1f %7.1f %7.1f %7.3f  %7.5f  %7.3f" %summary_data, file=file, flush=True)


    # Flow Stations
    """fs_names = ['fc.Fl_O','inlet.Fl_O','fan.Fl_O','splitter.Fl_O1','duct4.Fl_O',
                'lpc.Fl_O','duct6.Fl_O','hpc.Fl_O','bld3.Fl_O',
                'burner.Fl_O','hpt.Fl_O','duct11.Fl_O','lpt.Fl_O','duct13.Fl_O',
                'core_nozz.Fl_O','splitter.Fl_O2','byp_bld.Fl_O','duct15.Fl_O',
                'byp_nozz.Fl_O']"""

    fs_names = ['fc.Fl_O','inlet.Fl_O','fan.Fl_O','splitter.Fl_O2','splitter.Fl_O1','byp_bld.Fl_O', 'duct4.Fl_O',
                'lpc.Fl_O','duct6.Fl_O','hpc.Fl_O','bld3.Fl_O',
                'burner.Fl_O','hpt.Fl_O','duct11.Fl_O','lpt.Fl_O','duct13.Fl_O',
                'core_nozz.Fl_O','duct15.Fl_O', 'byp_nozz.Fl_O']
    fs_full_names = [f'{pt}.{fs}' for fs in fs_names]
    pyc.print_flow_station(prob, fs_full_names, file=file)

    # Compressor Names
    comp_names = ['fan', 'lpc', 'hpc']
    comp_full_names = [f'{pt}.{c}' for c in comp_names]
    pyc.print_compressor(prob, comp_full_names, file=file)

    # Burner 
    pyc.print_burner(prob, [f'{pt}.burner'])

    # Turbine Names
    turb_names = ['hpt', 'lpt']
    turb_full_names = [f'{pt}.{t}' for t in turb_names]
    pyc.print_turbine(prob, turb_full_names, file=file)

    # Nozzle Names 
    noz_names = ['core_nozz', 'byp_nozz']
    noz_full_names = [f'{pt}.{n}' for n in noz_names]
    pyc.print_nozzle(prob, noz_full_names, file=file)

    # Shaft Names 
    shaft_names = ['hp_shaft', 'lp_shaft', 'fan_shaft']
    shaft_full_names = [f'{pt}.{s}' for s in shaft_names]
    pyc.print_shaft(prob, shaft_full_names, file=file)

    # Bleed Names 
    bleed_names = ['hpc','bld3','byp_bld']
    bleed_full_names = [f'{pt}.{b}' for b in bleed_names]
    pyc.print_bleed(prob, bleed_full_names, file=file)


class MPPW1133(pyc.MPCycle):

    def initialize(self):
        self.options.declare('order_add', default=[],
                              desc='Name of subsystems to add to end of order.')
        self.options.declare('order_start', default=[],
                              desc='Name of subsystems to add to beginning of order.')
        self.options.declare('statics', default=True,
                              desc='Tells the model whether or not to connect areas.')

        super().initialize()

    def setup(self):

        # CRZ POINT (DESIGN)
        self.pyc_add_pnt('CRZ', PW1133(), promotes_inputs=[('fan.PR', 'fan:PRdes'), ('lpc.PR', 'lpc:PRdes'), 
                                                        ('opr_calc.FPR', 'fan:PRdes'), ('opr_calc.LPCPR', 'lpc:PRdes'), ('hpt.PR', 'hpt:PRdes'), ('lpt.PR', 'lpt:PRdes')])

        # POINT 1: Cruise (CRZ)
        self.set_input_defaults('CRZ.fc.alt', 35000., units='ft'),
        self.set_input_defaults('CRZ.fc.MN', 0.78),
        self.set_input_defaults('CRZ.inlet.ram_recovery', 0.999),

        self.set_input_defaults('CRZ.burner.Fl_I:FAR', 0.02672),

        self.set_input_defaults('CRZ.hpc.eff', 0.8819),

        self.set_input_defaults('CRZ.balance.rhs:fan_eff', 0.9286),
        self.set_input_defaults('CRZ.duct4.dPqP', 0.0048),
        self.set_input_defaults('CRZ.balance.rhs:lpc_eff', 0.9150),
        self.set_input_defaults('CRZ.duct6.dPqP', 0.0101),
        self.set_input_defaults('CRZ.balance.rhs:hpt_eff', 0.8734), 
        self.set_input_defaults('CRZ.duct11.dPqP', 0.0051),
        self.set_input_defaults('CRZ.balance.rhs:lpt_eff', 0.8854), 
        self.set_input_defaults('CRZ.duct13.dPqP', 0.0107),
        self.set_input_defaults('CRZ.duct15.dPqP', 0.0136),
        self.set_input_defaults('CRZ.Fan_Nmech', 1559, units='rpm'),
        self.set_input_defaults('CRZ.LP_Nmech', 4833, units='rpm'),
        self.set_input_defaults('CRZ.HP_Nmech', 13500, units='rpm'),
        
        # Air Stream Velocity
        self.set_input_defaults('CRZ.inlet.MN', 0.6757),
        self.set_input_defaults('CRZ.fan.MN', 0.4108)
        self.set_input_defaults('CRZ.splitter.MN1', 0.3098)
        self.set_input_defaults('CRZ.splitter.MN2', 0.4231)
        self.set_input_defaults('CRZ.duct4.MN', 0.3115),
        self.set_input_defaults('CRZ.lpc.MN', 0.2665),
        self.set_input_defaults('CRZ.duct6.MN', 0.3563),
        self.set_input_defaults('CRZ.hpc.MN', 0.1801),
        self.set_input_defaults('CRZ.bld3.MN', 0.1801)
        self.set_input_defaults('CRZ.burner.MN', 0.0754),
        self.set_input_defaults('CRZ.hpt.MN', 0.3650),
        self.set_input_defaults('CRZ.duct11.MN', 0.3819),
        self.set_input_defaults('CRZ.lpt.MN', 0.5000),
        self.set_input_defaults('CRZ.duct13.MN', 0.5466),
        self.set_input_defaults('CRZ.byp_bld.MN', 0.4231),
        self.set_input_defaults('CRZ.duct15.MN', 0.4304),

        # Bleed Mass Flow, Pressure Drop, Work Fraction
        self.pyc_add_cycle_param('burner.dPqP', 0.025),
        self.pyc_add_cycle_param('core_nozz.Cv', 0.9905),
        self.pyc_add_cycle_param('byp_nozz.Cv', 0.9948),
        self.pyc_add_cycle_param('lp_shaft.fracLoss', 0.01)
        self.pyc_add_cycle_param('hp_shaft.HPX', 100.0, units='hp'),
        self.pyc_add_cycle_param('hpc.bld_inlet:frac_W', 0.045169),
        self.pyc_add_cycle_param('hpc.bld_inlet:frac_P', 0.5),
        self.pyc_add_cycle_param('hpc.bld_inlet:frac_work', 0.5),
        self.pyc_add_cycle_param('hpc.bld_exit:frac_W', 0.002612),
        self.pyc_add_cycle_param('hpc.bld_exit:frac_P', 0.5),
        self.pyc_add_cycle_param('hpc.bld_exit:frac_work', 0.5),
        self.pyc_add_cycle_param('hpc.cust:frac_W', 0.045133),
        self.pyc_add_cycle_param('hpc.cust:frac_P', 0.5),
        self.pyc_add_cycle_param('hpc.cust:frac_work', 0.5),
        self.pyc_add_cycle_param('bld3.bld_inlet:frac_W', 0.03731),
        self.pyc_add_cycle_param('bld3.bld_exit:frac_W', 0.06108),
        self.pyc_add_cycle_param('hpt.bld_inlet:frac_P', 1.0), #1.0
        self.pyc_add_cycle_param('hpt.bld_exit:frac_P', 0.0), #0.0
        self.pyc_add_cycle_param('lpt.bld_inlet:frac_P', 1.0), #1.0
        self.pyc_add_cycle_param('lpt.bld_exit:frac_P', 0.0), #0.0
        self.pyc_add_cycle_param('byp_bld.bypBld:frac_W', 0.0),

        # OTHER POINTS (OFF-DESIGN)
        self.od_pts = ['RTO','SLS','TOC']
        self.cooling = [True, False, False]
        self.od_MNs = [0.25, 0.000001, 0.85]
        self.od_alts = [0.0, 0.0, 39000.0]
        self.od_dTs = [27.0, 27.0, 0.0]
        self.od_BPRs = [1.6697, 1.5828, 2.3146]  #1.75, 1.75, 1.9397
        self.od_recoveries = [0.9926, 0.9925, 0.9991]

        for i, pt in enumerate(self.od_pts):
            self.pyc_add_pnt(pt, PW1133(design=False, cooling=self.cooling[i]))

            self.set_input_defaults(pt+'.fc.MN', val=self.od_MNs[i])
            self.set_input_defaults(pt+'.fc.alt', val=self.od_alts[i], units='ft')
            self.set_input_defaults(pt+'.fc.dTs', val=self.od_dTs[i], units='degR')
            self.set_input_defaults(pt+'.balance.rhs:BPR', val=self.od_BPRs[i])
            self.set_input_defaults(pt+'.inlet.ram_recovery', val=self.od_recoveries[i])

        # Extra set input for Rolling Takeoff
        self.set_input_defaults('RTO.balance.rhs:FAR', 25400.0, units='lbf'), 

        # Connection Off Design
        self.pyc_connect_des_od('fan.s_PR', 'fan.s_PR')
        self.pyc_connect_des_od('fan.s_Wc', 'fan.s_Wc')
        self.pyc_connect_des_od('fan.s_eff', 'fan.s_eff')
        self.pyc_connect_des_od('fan.s_Nc', 'fan.s_Nc')
        self.pyc_connect_des_od('lpc.s_PR', 'lpc.s_PR')
        self.pyc_connect_des_od('lpc.s_Wc', 'lpc.s_Wc')
        self.pyc_connect_des_od('lpc.s_eff', 'lpc.s_eff')
        self.pyc_connect_des_od('lpc.s_Nc', 'lpc.s_Nc')
        self.pyc_connect_des_od('hpc.s_PR', 'hpc.s_PR')
        self.pyc_connect_des_od('hpc.s_Wc', 'hpc.s_Wc')
        self.pyc_connect_des_od('hpc.s_eff', 'hpc.s_eff')
        self.pyc_connect_des_od('hpc.s_Nc', 'hpc.s_Nc')
        self.pyc_connect_des_od('hpt.s_PR', 'hpt.s_PR')
        self.pyc_connect_des_od('hpt.s_Wp', 'hpt.s_Wp')
        self.pyc_connect_des_od('hpt.s_eff', 'hpt.s_eff')
        self.pyc_connect_des_od('hpt.s_Np', 'hpt.s_Np')
        self.pyc_connect_des_od('lpt.s_PR', 'lpt.s_PR')
        self.pyc_connect_des_od('lpt.s_Wp', 'lpt.s_Wp')
        self.pyc_connect_des_od('lpt.s_eff', 'lpt.s_eff')
        self.pyc_connect_des_od('lpt.s_Np', 'lpt.s_Np')

        self.pyc_connect_des_od('gearbox.gear_ratio', 'gearbox.gear_ratio')
        self.pyc_connect_des_od('core_nozz.Throat:stat:area','balance.rhs:W')

        if self.options['statics'] is True:
            self.pyc_connect_des_od('inlet.Fl_O:stat:area', 'inlet.area')
            self.pyc_connect_des_od('fan.Fl_O:stat:area', 'fan.area')
            self.pyc_connect_des_od('splitter.Fl_O1:stat:area', 'splitter.area1')
            self.pyc_connect_des_od('splitter.Fl_O2:stat:area', 'splitter.area2')
            self.pyc_connect_des_od('duct4.Fl_O:stat:area', 'duct4.area')
            self.pyc_connect_des_od('lpc.Fl_O:stat:area', 'lpc.area')
            self.pyc_connect_des_od('duct6.Fl_O:stat:area', 'duct6.area')
            self.pyc_connect_des_od('hpc.Fl_O:stat:area', 'hpc.area')
            self.pyc_connect_des_od('bld3.Fl_O:stat:area', 'bld3.area')
            self.pyc_connect_des_od('burner.Fl_O:stat:area', 'burner.area')
            self.pyc_connect_des_od('hpt.Fl_O:stat:area', 'hpt.area')
            self.pyc_connect_des_od('duct11.Fl_O:stat:area', 'duct11.area')
            self.pyc_connect_des_od('lpt.Fl_O:stat:area', 'lpt.area')
            self.pyc_connect_des_od('duct13.Fl_O:stat:area', 'duct13.area')
            self.pyc_connect_des_od('byp_bld.Fl_O:stat:area', 'byp_bld.area')
            self.pyc_connect_des_od('duct15.Fl_O:stat:area', 'duct15.area')

        self.pyc_connect_des_od('duct4.s_dPqP', 'duct4.s_dPqP')
        self.pyc_connect_des_od('duct6.s_dPqP', 'duct6.s_dPqP')
        self.pyc_connect_des_od('duct11.s_dPqP', 'duct11.s_dPqP')
        self.pyc_connect_des_od('duct13.s_dPqP', 'duct13.s_dPqP')
        self.pyc_connect_des_od('duct15.s_dPqP', 'duct15.s_dPqP')

        # Bleed Connections
        #self.connect('RTO.hpt_chrg_cool_frac', 'CRZ.bld3.bld_exit:frac_W')
        #self.connect('RTO.hpt_nochrg_cool_frac', 'CRZ.bld3.bld_inlet:frac_W')

        #self.connect('RTO.hpt_chrg_cool_frac', 'SLS.bld3.bld_exit:frac_W')
        #self.connect('RTO.hpt_nochrg_cool_frac', 'SLS.bld3.bld_inlet:frac_W')

        #self.connect('RTO.hpt_chrg_cool_frac', 'TOC.bld3.bld_exit:frac_W')
        #self.connect('RTO.hpt_nochrg_cool_frac', 'TOC.bld3.bld_inlet:frac_W')

        self.add_subsystem('T4_ratio',
                             om.ExecComp('CRZ_T4 = RTO_T4*TR',
                                         RTO_T4={'val': 3380.0, 'units':'degR'},
                                         CRZ_T4={'val': 2994.5, 'units':'degR'},
                                         TR={'val': 0.885946, 'units': None}), promotes_inputs=['RTO_T4',])
        #Removed Line
        #self.connect('T4_ratio.CRZ_T4', 'CRZ.balance.rhs:FAR')
        initial_order = ['T4_ratio', 'CRZ', 'RTO', 'SLS', 'TOC']
        self.set_order(self.options['order_start'] + initial_order + self.options['order_add'])

        # Solvers Set Up
        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['atol'] = 1e-6
        newton.options['rtol'] = 1e-6
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 30
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 10
        newton.options['err_on_non_converge'] = True
        newton.options['reraise_child_analysiserror'] = False
        newton.linesearch =  om.BoundsEnforceLS()
        newton.linesearch.options['bound_enforcement'] = 'scalar'
        newton.linesearch.options['iprint'] = -1

        self.linear_solver = om.DirectSolver(assemble_jac=True)

        super().setup()

def PW1133ref_model():

    prob = om.Problem()

    prob.model = MPPW1133()

    # Comment out the optimization setup
    # Setup the optimization
    #prob.driver = om.ScipyOptimizeDriver()
    #prob.driver.options['optimizer'] = 'SLSQP'
    #prob.driver.options['debug_print'] = ['desvars', 'nl_cons', 'objs']
    #prob.driver.opt_settings={'Major step limit': 0.05}

    prob.model.add_design_var('fan:PRdes', lower=1.20, upper=1.8)
    #prob.model.add_design_var('lpc:PRdes', lower=1.5, upper=4.0)
    #prob.model.add_design_var('CRZ.balance.rhs:hpc_PR', lower=40.0, upper=70.0, ref0=40.0, ref=70.0)
    prob.model.add_design_var('RTO_T4', lower=3000.0, upper=3600.0, ref0=3000.0, ref=3600.0)
    prob.model.add_design_var('T4_ratio.TR', lower=0.5, upper=0.95, ref0=0.5, ref=0.95)

    # Comment out the objective
    #prob.model.add_objective('CRZ.perf.TSFC')

    # to add the constraint to the model
    prob.model.add_constraint('CRZ.fan_dia.FanDia', upper=100.0, ref=100.0)

    return(prob)

if __name__ == "__main__":

    import time

    prob = PW1133ref_model()

    prob.setup()

    # Define the design point
    prob.set_val('CRZ.fc.W', 556.24, units='lbm/s')
    prob.set_val('CRZ.splitter.BPR', 11.5523),
    prob.set_val('CRZ.balance.rhs:hpc_PR', 46.088)

    # Set up the specific cycle parameters
    prob.set_val('fan:PRdes', 1.52),
    prob.set_val('lpc:PRdes', 2.246),
    prob.set_val('hpt:PRdes', 3.963),
    prob.set_val('lpt:PRdes', 8.981),
    prob.set_val('T4_ratio.TR', 0.885946) #0.926470588
    prob.set_val('RTO_T4', 3380.0, units='degR')
    prob.set_val('SLS.balance.rhs:FAR', 33110 , units='lbf') 
    prob.set_val('TOC.balance.rhs:FAR', 5800, units='lbf')
    prob.set_val('RTO.hpt_cooling.x_factor', 0.9)

    # Set initial guesses for balances

    #prob['CRZ.balance.FAR'] = 0.02672
    #prob['CRZ.balance.lpt_PR'] = 8.981
    #prob['CRZ.balance.hpt_PR'] = 3.964
    prob['CRZ.fc.balance.Pt'] = 5.166
    prob['CRZ.fc.balance.Tt'] = 441.74

    FAR_guess = [0.03090, 0.03103, 0.02809]
    W_guess = [1286.29, 1212.70, 492.08]
    BPR_guess = [11.8101, 11.4680, 11.3464]
    fan_Nmech_guess = [1557.6, 1550.2, 1617.0]
    lp_Nmech_guess = [4828.60, 4805.68 , 5012.74]
    hp_Nmech_guess = [14622.34, 14576.6, 13689.08]
    hpt_PR_guess = [3.963, 3.965, 3.953]
    lpt_PR_guess = [7.249, 7.114, 9.072]
    fan_Rline_guess = [1.6697, 1.5828, 2.3146]
    lpc_Rline_guess = [1.9402, 2.0043, 2.4761]
    hpc_Rline_guess = [2.0336, 2.0317, 2.0588]
    trq_guess = [52509.1, 41779.4, 22369.7]

    for i, pt in enumerate(prob.model.od_pts):

        # initial guesses
        prob[pt+'.balance.FAR'] = FAR_guess[i]
        prob[pt+'.balance.W'] = W_guess[i]
        prob[pt+'.balance.BPR'] = BPR_guess[i]
        prob[pt+'.balance.fan_Nmech'] = fan_Nmech_guess[i]
        prob[pt+'.balance.lp_Nmech'] = lp_Nmech_guess[i]
        prob[pt+'.balance.hp_Nmech'] = hp_Nmech_guess[i]
        prob[pt+'.hpt.PR'] = hpt_PR_guess[i]
        prob[pt+'.lpt.PR'] = lpt_PR_guess[i]
        prob[pt+'.fan.map.RlineMap'] = fan_Rline_guess[i]
        prob[pt+'.lpc.map.RlineMap'] = lpc_Rline_guess[i]
        prob[pt+'.hpc.map.RlineMap'] = hpc_Rline_guess[i]
        prob[pt+'.gearbox.trq_base'] = trq_guess[i]

    st = time.time()

    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=1)
    prob.run_model()

    for pt in ['CRZ']+prob.model.od_pts:
        viewer(prob, pt)

    print("time", time.time() - st)