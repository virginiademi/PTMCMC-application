"Code for calculating magnetic field from given dipole moments, and  then calculating chi^2 from a given data."


import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize

from copy import deepcopy

npfuncs = {}
npfuncs['pi'] = np.pi
npfuncs['abs'] = np.abs
npfuncs['sum'] = np.sum
npfuncs['array'] = np.array
npfuncs['size'] = np.size
npfuncs['sqrt'] = np.sqrt
npfuncs['sindeg'] = lambda thetadeg: np.sin(np.pi*thetadeg/180)
npfuncs['cosdeg'] = lambda thetadeg: np.cos(np.pi*thetadeg/180)

# npfuncs['stack'] = np.stack
# npfuncs['ones']  = np.ones

def ReadAndParseLocation(loc):
    # Give a dict containing coordinate system and a tuple of values for the vector in that coordinate system
    # Works in 3 D if its a 3-tuple
    # This function is used by other functions and most times not directly by the user. 
    # The user should be able to input all locations in a dict with coordinate type and location as a tuple.
    
    if loc['coordinate'] == 'spherical':
        loc_return = {'coordinate':'spherical'}
        r = loc['location'][0]
        theta = loc['location'][1]
        loc_return['location'] = {'r':r, 'theta':theta}
        if len(loc['location'])>2:
            phi = loc['location'][2]
            loc_return['location']['phi'] = phi
    elif loc['coordinate'] =='polar':
        loc_return = {'coordinate':'polar'}
        r = loc['location'][0]
        theta = loc['location'][1]
        loc_return['location'] = {'r':r, 'theta':theta}
        if len(loc['location'])>2:
            z = loc['location'][2]
            loc_return['location']['z'] = z
    elif loc['coordinate'] == 'cartesian':
        loc_return = {'coordinate':'cartesian'}
        x = loc['location'][0]
        y = loc['location'][1]
        loc_return['location'] = {'x':x, 'y':y}
        if len(loc['location'])>2:
            z = loc['location'][2]
            loc_return['location']['z'] = z
    else:
        print('Location coordinate must be either spherical (r,theta,phi), polar (r,theta,z), or cartesian (x,y,z)')
    return loc_return


def GetCartesian(loc_in):
    loc = ReadAndParseLocation(loc_in)
    loc_cartesian = {'coordinate':'cartesian'}
    if loc['coordinate'] == 'spherical':
        r = loc['location']['r']
        theta = loc['location']['theta']
        if 'phi' in loc['location']:
            phi = loc['location']['phi']        
            loc_cartesian['location'] = r([npfuncs['cosdeg'](theta)*npfuncs['cosdeg'](phi),
                                    npfuncs['sindeg'](theta)*npfuncs['cosdeg'](phi),
                                    npfuncs['sindeg'](phi)])
        else:
            loc_cartesian['location'] = r([npfuncs['cosdeg'](theta),
                                    npfuncs['sindeg'](theta)])
    elif loc['coordinate'] == 'polar':
        r = loc['location']['r']
        theta = loc['location']['theta']
        if 'z' in  loc['location']:
            z = loc['location']['z']
            loc_cartesian['location'] = ([r*npfuncs['cosdeg'](theta),
                                    r*npfuncs['sindeg'](theta),
                                    z])
        else:
            
            loc_cartesian['location'] = ([r*npfuncs['cosdeg'](theta),
                                    r*npfuncs['sindeg'](theta)])
    elif loc['coordinate'] == 'cartesian':
        if 'z' in loc['location']:
            loc_cartesian['location'] = [loc['location']['x'],
                                         loc['location']['y'],
                                         loc['location']['z']]
        else:
            loc_cartesian['location'] = [loc['location']['x'],loc['location']['y']]
                
    else:
        print('Location coordinate must be either spherical (r,theta,phi), polar (r,theta,z), or cartesian (x,y,z)')
    return loc_cartesian

def GetPolar(loc_in):
    loc = ReadAndParseLocation(loc_in)
    loc_polar={'coordinate':'polar'}
    if loc['coordinate'] == 'spherical':
        if 'phi' in loc['location']:
            loc_polar['location'] = [loc['location']['r']*npfuncs['cosdeg'](loc['location']['phi']),
                         loc['location']['theta'],
                         loc['location']['r']*npfuncs['sindeg'](loc['location']['phi'])]
        else:
            loc_polar['location'] = [loc['location']['r'],loc['location']['theta']]
    elif loc['coordinate'] =='polar':
        if 'z' in loc['location']:
            loc_polar['location'] = [loc['location']['r'],
                         loc['location']['theta'],
                         loc['location']['z']]
        else:
            loc_polar['location'] = [loc['location']['r'],loc['location']['theta']]
    elif loc['coordinate'] == 'cartesian':
        if 'z' in loc['location']:
            loc_polar['location'] = [npfuncs['sqrt'](loc['location']['x']**2+loc['location']['y']**2),
                                    npfuncs['atan2'](loc['location']['y'],loc['location']['x']),
                                    loc['location']['z']]
        else:
            loc_polar['location'] = [npfuncs['sqrt'](loc['location']['x']**2+loc['location']['y']**2),
                                    npfuncs['atan2'](loc['location']['y'],loc['location']['x'])]
    else:
        print('Location coordinate must be either spherical (r,theta,phi), polar (r,theta,z), or cartesian (x,y,z)')
    return loc_polar




def CheckValid(source_location_in,rotor_dimensions):
    # First check if source lies inside the volume:
    valid_basic = 0
    valid = 0
    source_location = GetPolar(source_location_in)

    if len(source_location['location'])>2: # In cases when z is being used
        if ((2*npfuncs['abs'](source_location['location'][2])<rotor_dimensions['height']) and 
            (source_location['location'][0]<rotor_dimensions['outer_radius'])and 
            (source_location['location'][0]>rotor_dimensions['hole_radius'])):
            # This now means the thing is inside the rotor boundary
            valid_basic = 1
#             print('Source matches z and r')
#            print(valid)
    else: # In cases when z is not being used
        if source_location['location'][0]<rotor_dimensions['outer_radius']:
            valid_basic = 1
#             print('no z given, source matches r')
#     print(source_location)
    # remove negative r
    if source_location['location'][0]<0:
        valid_basic  = 0
#         print('r negative')
    # remove theta outside 2pi
    if source_location['location'][1]>360:
        valid_basic = 0
#         print('theta greater than $2\pi$')
    if source_location['location'][1]<0:
#         print('theta negative')
        valid_basic = 0
    
#     print('Basic validity  = %0.0f'%(valid_basic))
    # Now let's check if the source is in the rotor's anulus or bar
    if valid_basic:
        # Check if its in the anulus
        if source_location['location'][0]>rotor_dimensions['inner_radius']:
            valid = 1
        # If not, check if its in the bar
        elif npfuncs['abs'](2*source_location['location'][0]*
                            npfuncs['sindeg'](source_location['location'][1]))<rotor_dimensions['bar_width']:
            valid = 1
#     print('Final validity  = %0.0f'%(valid))
    return valid

def chi2_one_exp_one_freq(optimization_parameters,optimization_settings):
    # For scipy minimize to work, optimization_parameters need to be a 1-D array.
    # Rest of the arguments can be whatever data structures.
    
    # This function will take all the variables that are being optimized, 
    # index to specify things about the optimization,
    # experimental parameters to say about the experiment
    # Enter all lengths in mm, all angles in degrees, all magnetic moments in 1e-11 SI, all DC offsets in pT
    # Let's start by unpacking all the data
    # Find number of sources
    n_sources = optimization_settings['number of sources']
    n_sensors = optimization_settings['number of sensors']
    loc_dim = optimization_settings['location dimensions']
    m_dim = optimization_settings['moment dimensions']
    input_params_for_chi = deepcopy(optimization_settings)
    if 'bar location' in optimization_settings:
        n_bar = 0
    elif optimization_settings['optimize bar location']:
        n_bar = 1
        input_params_for_chi['bar location'] = optimization_parameters[-1]
        if input_params_for_chi['bar location']<0:
            print('Bar must be located from 0 to 180 degrees')
            return 1000
        if input_params_for_chi['bar location']>180:
            print('Bar must be located from 0 to 180 degrees')
            return 1000
    else:
        input_params_for_chi['bar location'] = 0
            
    anticipated_size = n_sensors*2 + n_sources*(loc_dim+m_dim) + n_bar
    if len(optimization_parameters)<anticipated_size:
        print('insufficient parameters')
        return 1000
    elif len(optimization_parameters)>anticipated_size:
        print('too many parameters')
        return 1000
    print(optimization_parameters)
    # now first loc_dim + m_dim elements are for first source, and so on
    source_locations = []
    source_moments = []
    ii = 0
    i_src = 0
    while ii+1 < n_sources*(loc_dim+m_dim):
        if loc_dim>2:
            src_loc = {'coordinate': optimization_settings['location coordinate system'],
                       'location':optimization_parameters[ii:ii+loc_dim]}
        else:
            src_loc = {'coordinate': optimization_settings['location coordinate system'],
                       'location':list(optimization_parameters[ii:ii+loc_dim])+[0]}
#             print(src_loc)
        
        source_locations.append(src_loc)
        if m_dim>2:
            src_m = {'coordinate':optimization_settings['moment coordinate system'],
                'moment':optimization_parameters[ii+loc_dim:ii+loc_dim+m_dim]}
        else:
            src_m = {'coordinate':optimization_settings['moment coordinate system'],
                'moment':list(optimization_parameters[ii+loc_dim:ii+loc_dim+m_dim])+[0]}
        source_moments.append(src_m)
        ii += loc_dim+m_dim
    # Create a new packaged data structure to calculate B field
#     print(optimization_settings)
    
#     print(input_params_for_B)
    input_params_for_chi['source locations']=source_locations
    input_params_for_chi['source moments']=source_moments
    input_params_for_chi['DC offsets'] = optimization_parameters[ii:ii+n_sensors*2]
#     print(input_params_for_B)
#     print(optimization_settings)
    if optimization_settings['plot']:
        plt.figure()
        ax = plt.axes()
    else:
        ax = 0
    if optimization_settings['rotation sign']!=0: # ie the sign is known and not needed to optimize
        omega = 2*npfuncs['pi']*optimization_settings['rotation sign']*optimization_settings['rotation frequency']
        # Call a function here that gives magnetic field as a function of time, locs, m and frequency
        input_params_for_chi['omega'] = omega
        chi  = calc_chi_single_freq(input_params_for_chi,ax)
    else:
        # Try both signs and see which one gives lower chi
        omega_pos = 2*npfuncs['pi']*optimization_settings['rotation frequency']
        input_params_for_chi['omega'] = omega_pos
        # Call a function here that gives magnetic field as a function of time, locs, m and frequency
        chi_pos = calc_chi_single_freq(input_params_for_chi,ax)
        omega_neg = -2*npfuncs['pi']*optimization_settings['rotation frequency']
        input_params_for_chi['omega'] = omega_neg
        # Call a function here that gives magnetic field as a function of time, locs, m and frequency
        chi_neg = calc_chi_single_freq(input_params_for_chi,ax)
        chi = min(chi_pos,chi_neg)
    
    ## Plotting
    if optimization_settings['plot']:
        textstring = """ $\chi^2 = $ %0.2e 
    Dipole #               Position                              Moment""" %(chi)
        for i_source in range(n_sources):
            source_loc = deepcopy(input_params_for_chi['source locations'][i_source])
            source_loc_polar = GetPolar(source_loc)
            source_m = input_params_for_chi['source moments'][i_source]
    #       source_m_polar = GetPolar(source_m)
            strloop = """
            %0.0f      (%0.1f mm, %0.1f deg, %0.1f mm)      (%0.1f,%0.1f,%0.1f)* 1e-11 A/$\mathrm{mm}^2$  """\
            %(i_source,*source_loc_polar['location'],*source_m['moment'])
            textstring = textstring+strloop
        ax.legend()
        ax.grid()
    #        axispos = plt.gca().get_position()
    #        print(axispos)
        xmin,xmax = ax.get_xlim()
        ymin,ymax = ax.get_ylim()
        xloc = xmin - 0.12*(xmax-xmin)
        yloc = ymax + 0.05*(ymax-ymin)
        ax.text(xloc,yloc,textstring)
    #         print(xloc)
    #         print(yloc)
        ax.set_xlabel('theta (units of $\pi$)')
        ax.set_ylabel('Magnetic field (pT)')
        if optimization_settings['saveplot']:
            plt.savefig(optimization_settings['figname']+'.png',bbox_inches='tight',dpi = 300)
#         plt.figure()
#         ax2 = plt.axes()
        locfig = make_position_figure(input_params_for_chi)
        if optimization_settings['saveplot']:
            locfig.savefig(optimization_settings['figname']+'_locations.png',bbox_inches='tight',dpi = 300)

    return chi

def calc_chi_single_freq(input_params,pltax):
    
    # Enter all lengths in mm, all angles in degrees, all magnetic moments in 1e-11 SI, all DC offsets in pT
    
    # Get B data
    B_data = npfuncs['array'](input_params['B data'])
    B_DC = 1e-12*npfuncs['array'](input_params['DC offsets'])
    
    
    # Calculate B from sources
    
    time = input_params['time']
    omega = input_params['omega']
    theta_rotation = 180*omega*time/npfuncs['pi']
    n_sensors = input_params['number of sensors']
    n_sources = input_params['number of sources']
    
    B_calc_AC = 0*B_data
    row = 0
    for j_sensor in range(n_sensors):
        sensor_loc = input_params['sensor locations'][j_sensor]
        sensor_direction = input_params['sensor field order'][j_sensor]
        B_j = np.zeros((len(sensor_direction),len(theta_rotation)))
#         theta_current_source = 0
        for i_source in range(n_sources):
            source_m = deepcopy(input_params['source moments'][i_source])
#             print(source_m)
            source_loc = deepcopy(input_params['source locations'][i_source])
#             print('before: sensor = %0.0f, source =%0.0f, theta_source = %0.0f,theta_cumulative=%0.0f'
#                   %(j_sensor,i_source,source_loc['location'][1],theta_current_source))
            if source_loc['location'][1]>360:
                print('Source %0.0f has theta bigger than 2pi'%(i_source+1))
                return 1000
            if source_loc['location'][1]<0:
                print('Source %0.0f has theta negative'%(i_source+1))
                return 1000
#             theta_current_source += source_loc['location'][1]
#             if theta_current_source>360:
#                 theta_current_source = theta_current_source%360
#             # Each source's angular location is expected to be given relative to the source before it.
#             # So we add all the previous ones to get the physical location.
#             # This helps reduce the number of parameters, while keeping the parameters easily visualizable.
            
#             source_loc['location'][1] = theta_current_source
# #             print('after: sensor = %0.0f, source =%0.0f, theta_source = %0.0f,theta_cumulative=%0.0f'
# #                   %(j_sensor,i_source,source_loc['location'][1],theta_current_source))

            ## Check whether this current source is in the rotor
            val = CheckValid(source_loc,input_params['rotor dimensions'])
            if val!=1:
                print('Source %0.0f not in rotor'%(i_source+1))
#                print(src_loc)
                return 1000
#           else:
#                print('Source in rotor, proceeding')
            
            
            
            # Set this "actual angular location" before passing it to the function that calculates field.
            B_i_full = CalcB(source_loc,source_m,sensor_loc,theta_rotation+input_params['bar location'])
            B_i = B_i_full[sensor_direction,:]
            B_j += B_i
            del source_m
            del source_loc
        B_calc_AC[row:row+len(sensor_direction),:] = B_j
        row = row + len(sensor_direction)
    B_calc = (B_calc_AC.T + B_DC).T
#         print(j_sensor)
#         print(row)
#         print(B_calc)
#     B_calc = np.array(B_calc)
#     print(B_calc)
#     print(B_calc.shape)
    # Calculate chi
            
    chi_sq = npfuncs['sum'](((B_data-B_calc)/(1e-12))**2)
    chi_reduced = (chi_sq)/(B_data.size**2) # The size function is called on the object, so the npfuncs['size'] didn't work. If this turned out to be slow, another way could be found.
    print(chi_reduced)

    if input_params['plot']:
#         plt.figure()
        
        for p in range(row):
            if 'plotlabels' in input_params:
                h = pltax.plot(theta_rotation/180,1e12*B_data[p],
                             label = input_params['plotlabels'][p]+' '+str(omega/npfuncs['pi']/2)+' Hz')
            else:
                h = pltax.plot(theta_rotation/180,1e12*B_data[p],label=p+' '+str(omega/npfuncs['pi']/2)+' Hz')
            pltax.plot(theta_rotation/180,1e12*B_calc[p],color = h[-1].get_color(),linestyle = '--')
            
    return chi_reduced

def CalcB(source_loc,source_m,sensor_loc,theta_rotation):
    # This function will take a single source, a single sensor (everything 3D), and give a 3 dimensional magnetic field at the sensor due to the source as a function of rotation
    # Enter all lengths in mm, all angles in degrees, all magnetic moments in 1e-11 SI, all DC offsets in pT
    mu0_by_4pi = 1e-7
    # First get source parameters
    sensor_cartesian = GetCartesian(sensor_loc)['location']
    source_polar = GetPolar(source_loc)
    source_r = source_polar['location'][0]
    source_theta = source_polar['location'][1]
    source_z = source_polar['location'][2]
    # Now we make a rotation operator and multiply it to m and source location
    theta_rotation_plus_source = theta_rotation + source_theta
    R_theta = np.array([[[npfuncs['cosdeg'](th),-npfuncs['sindeg'](th),0],
                         [npfuncs['sindeg'](th),npfuncs['cosdeg'](th),0],
                         [0,0,1]] 
                                  for th in theta_rotation_plus_source])
    source_m_cartesian_theta = np.matmul(R_theta,source_m['moment']).transpose()
    source_r_cartesian_theta = np.matmul(R_theta,[source_r,0,source_z]).transpose()
#     print(sensor_cartesian)
#     print(source_m_cartesian_theta)
#     print(source_r_cartesian_theta)
    d_vec = (source_r_cartesian_theta.transpose()-sensor_cartesian).transpose()
    
    # Convert lengths to meters, and moments to SI
    d_vec = 1e-3*d_vec
    source_m_cartesian_theta = 1e-11*source_m_cartesian_theta
    
    #print(d_vec.shape)
    #plt.plot(theta,m_theta[0,:])
    #plt.plot(theta,m_theta[1,:])
    #plt.plot(theta,m_theta[2,:])
    mdotd = np.diagonal(np.matmul(d_vec.transpose(),source_m_cartesian_theta))
   # print(mdotd.shape)
    comp1 = 3*np.divide(np.multiply(d_vec,mdotd),np.linalg.norm(d_vec,axis=0)**2)
#    print(comp1.shape)
    B_vec = mu0_by_4pi/np.linalg.norm(d_vec,axis=0)**3 *(comp1-source_m_cartesian_theta)
#    print(B_vec.shape)
    
    return B_vec



def chi2_one_exp_multi_freq(optimization_parameters,optimization_settings):
    # For scipy minimize to work, optimization_parameters need to be a 1-D array.
    # Rest of the arguments can be whatever data structures.
    
    # This function will take all the variables that are being optimized, 
    # index to specify things about the optimization,
    # experimental parameters to say about the experiment
    # Enter all lengths in mm, all angles in degrees, all magnetic moments in 1e-11 SI, all DC offsets in pT
    # Let's start by unpacking all the data
    # Find number of sources
    n_freqs   = optimization_settings['number of frequencies']
    n_sources = optimization_settings['number of sources']
    n_sensors = optimization_settings['number of sensors']
    loc_dim = optimization_settings['location dimensions']
    m_dim = optimization_settings['moment dimensions']
    anticipated_size = n_sensors*2 + n_sources*(loc_dim+m_dim)
    if len(optimization_parameters)<anticipated_size:
        print('insufficient parameters')
        return 1000
    elif len(optimization_parameters)>anticipated_size:
        print('too many parameters')
        return 1000
    print(optimization_parameters)
    # now first loc_dim + m_dim elements are for first source, and so on
    source_locations = []
    source_moments = []
    ii = 0
    i_src = 0
    while ii+1 < n_sources*(loc_dim+m_dim):
        if loc_dim>2:
            src_loc = {'coordinate': optimization_settings['location coordinate system'],
                       'location':optimization_parameters[ii:ii+loc_dim]}
        else:
            src_loc = {'coordinate': optimization_settings['location coordinate system'],
                       'location':list(optimization_parameters[ii:ii+loc_dim])+[0]}
#             print(src_loc)
        
        source_locations.append(src_loc)
        if m_dim>2:
            src_m = {'coordinate':optimization_settings['moment coordinate system'],
                'moment':optimization_parameters[ii+loc_dim:ii+loc_dim+m_dim]}
        else:
            src_m = {'coordinate':optimization_settings['moment coordinate system'],
                'moment':list(optimization_parameters[ii+loc_dim:ii+loc_dim+m_dim])+[0]}
        source_moments.append(src_m)
        ii += loc_dim+m_dim
    # Create a new packaged data structure to calculate B field
#     print(optimization_settings)
    input_params_for_chi = deepcopy(optimization_settings)
#     print(input_params_for_B)
    input_params_for_chi['source locations']=source_locations
    input_params_for_chi['source moments']=source_moments
    input_params_for_chi['DC offsets'] = optimization_parameters[ii:ii+n_sensors*2]
#     print(input_params_for_B)
#     print(optimization_settings)
    chi_freq=[]
    if optimization_settings['plot']:
        plt.figure()
        ax = plt.axes()
    else:
        ax = 0
    for i_freq in range(n_freqs):
        input_params_for_chi['bar location'] = optimization_settings['bar location'][i_freq]
        input_params_for_chi['B data']       = optimization_settings['B data'][i_freq]
        input_params_for_chi['time']         = optimization_settings['time'][i_freq]
        if input_params_for_chi['saveplot']:
            input_params_for_chi['figname'] = optimization_settings['figname'][i_freq]
#         print(optimization_settings['rotation frequency'][i_freq])
        if optimization_settings['rotation sign']!=0: # ie the sign is known and not needed to optimize
#             print(optimization_settings['rotation frequency'][i_freq])
            omega = 2*npfuncs['pi']*optimization_settings['rotation sign']*optimization_settings['rotation frequency'][i_freq]
            # Call a function here that gives magnetic field as a function of time, locs, m and frequency
            input_params_for_chi['omega'] = omega
            chi_freq.append(calc_chi_single_freq(input_params_for_chi,ax))
        else:
            # Try both signs and see which one gives lower chi
            omega_pos = 2*npfuncs['pi']*optimization_settings['rotation frequency'][i]
            input_params_for_chi['omega'] = omega_pos
            # Call a function here that gives magnetic field as a function of time, locs, m and frequency
            chi_pos = calc_chi_single_freq(input_params_for_chi,ax)
            omega_neg = -2*npfuncs['pi']*optimization_settings['rotation frequency'][i]
            input_params_for_chi['omega'] = omega_neg
            # Call a function here that gives magnetic field as a function of time, locs, m and frequency
            chi_neg = calc_chi_single_freq(input_params_for_chi,ax)
            chi_freq.append(min(chi_pos,chi_neg))
    chi = sum(chi_freq)
    print('Total $\chi^2$ = %0.2f'%(chi))
    
    ## Plotting
    if optimization_settings['plot']:
        textstring = """ $\chi^2 = $ %0.2e 
    Dipole #               Position                         Moment""" %(chi)
        for i_source in range(n_sources):
            source_loc = deepcopy(input_params_for_chi['source locations'][i_source])
            source_loc_polar = GetPolar(source_loc)
            source_m = input_params_for_chi['source moments'][i_source]
    #       source_m_polar = GetPolar(source_m)
            strloop = """
            %0.0f      (%0.1f mm, %0.1f deg, %0.1f mm)      (%0.1f,%0.1f,%0.1f)* 1e-11 A/$\mathrm{mm}^2$  """\
            %(i_source,*source_loc_polar['location'],*source_m['moment'])
            textstring = textstring+strloop
        lgd = ax.legend(bbox_to_anchor=(1.35, 1.3))
        ax.grid()
    #        axispos = plt.gca().get_position()
    #        print(axispos)
        xmin,xmax = ax.get_xlim()
        ymin,ymax = ax.get_ylim()
        xloc = xmin - 0.12*(xmax-xmin)
        yloc = ymax + 0.05*(ymax-ymin)
        ax.text(xloc,yloc,textstring)
    #         print(xloc)
    #         print(yloc)
        ax.set_xlabel('theta (units of $\pi$)')
        ax.set_ylabel('Magnetic field (pT)')
        if optimization_settings['saveplot']:
            plt.savefig(optimization_settings['figname']+'_field.png' ,bbox_extra_artists=(lgd,),bbox_inches='tight',dpi = 700)
        ## Script to make rotor and put dipoles in it
#         plt.figure()
#         ax2 = plt.axes()
        locfig = make_position_figure(input_params_for_chi)
        if optimization_settings['saveplot']:
            locfig.savefig(optimization_settings['figname']+'_locations.png' ,bbox_inches='tight',dpi = 700)

        


    return chi


def make_position_figure(input_dict):
#     import matplotlib.gridspec as gridspec
#     gridspec.GridSpec(3,1)
    fig, [sp1,sp2] = plt.subplots(nrows=2, ncols=1, sharex='col' 
,                               gridspec_kw={'height_ratios': [3.5, 1]}
#                                   ,figsize = (7,8)
                              )
#     fig, [sp1,sp2] = plt.subplots(nrows=2, ncols=1, sharex='all'                      
#                               )
    

#     sp2 = plt.subplot2grid((3,1), (0,0),  rowspan=2)
    ## Script to make rotor and put dipoles in it
    th = np.arange(0,360,1)
    rotor_out_x = input_dict['rotor dimensions']['outer_radius']*npfuncs['cosdeg'](th)
    rotor_out_y = input_dict['rotor dimensions']['outer_radius']*npfuncs['sindeg'](th)
    rotor_in_x = input_dict['rotor dimensions']['inner_radius']*npfuncs['cosdeg'](th)
    rotor_in_y = input_dict['rotor dimensions']['inner_radius']*npfuncs['sindeg'](th)
    rotor_hole_x = input_dict['rotor dimensions']['hole_radius']*npfuncs['cosdeg'](th)
    rotor_hole_y = input_dict['rotor dimensions']['hole_radius']*npfuncs['sindeg'](th)
    
    rtr_color = 'magenta'

    h = sp1.plot(rotor_out_x,rotor_out_y,label = 'rotor boundary',color = rtr_color)
    sp1.plot(rotor_in_x,rotor_in_y,color = h[-1].get_color())
    sp1.plot(rotor_hole_x,rotor_hole_y,color = h[-1].get_color())
    bar_x = np.arange(-input_dict['rotor dimensions']['inner_radius'],input_dict['rotor dimensions']['inner_radius'],1)
    bar_y = input_dict['rotor dimensions']['bar_width']/2*np.ones(len(bar_x))
    sp1.plot(bar_x,bar_y,bar_x,-bar_y,color = h[-1].get_color())
    
    

    sp1.set_ylabel('y (mm)')
    sp2.set_xlabel('x (mm)')
        
#     sp2 =  plt.subplot2grid((3,1), (2,0),sharex=sp1)
    rotor_out_z = input_dict['rotor dimensions']['height']/2*np.ones(len(rotor_out_x))
    sp2.plot(rotor_out_x,rotor_out_z,rotor_out_x,-rotor_out_z,color = rtr_color)
    sp2.set_ylabel('z (mm)')
    
    vert_z = np.arange(-input_dict['rotor dimensions']['height']/2,input_dict['rotor dimensions']['height']/2,input_dict['rotor dimensions']['height']/20)
    vert_x = input_dict['rotor dimensions']['outer_radius']*np.ones(len(vert_z))
    sp2.plot(vert_x,vert_z,-vert_x,vert_z,color = rtr_color)
    
#     plt.xlabel('x')
    
    for i_sensor in range(input_dict['number of sensors']):
        sensor_cartesian = GetCartesian(input_dict['sensor locations'][i_sensor])
        hsens = sp1.scatter(sensor_cartesian['location'][0],sensor_cartesian['location'][1],label = 'sensor '+str(i_sensor+1))
        sp2.scatter(sensor_cartesian['location'][0],sensor_cartesian['location'][2],color = hsens.get_facecolor())
        
    for i_source in range(input_dict['number of sources']):
        source_cartesian = GetCartesian(input_dict['source locations'][i_source])
        hsrc = sp1.scatter(source_cartesian['location'][0],source_cartesian['location'][1],label = 'source ' + str(i_source+1))
        sp2.scatter(source_cartesian['location'][0],source_cartesian['location'][2],color = hsrc.get_facecolor())
#     sp1.axis('scaled')
#     sp2.axis('scaled')
#     xlim_sp1 = sp1.get_xlim()
#     xlim_sp2 = sp2.get_xlim()
#     ylim_sp1 = sp1.get_ylim()
#     xlim_sp2 = sp1.get_ylim()
    
    
#     sp1.axis([-30,30,-30,30])
#     sp2.axis([-30,30,-10,17])
#     sp1.axis('equal')
#     sp2.axis('equal')

    
    sp1.legend()
    
    return fig
                                                                     
                                                                    

#     plt.xlim([-30,30])
#     plt.ylim([-30,30])
 
