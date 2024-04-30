import matplotlib.pyplot as plt


def interplanetry_porkchip(config):
    _config={
        'planet0'       :'Earth',
        'planet1'       :'MARS_BARYCENTER',
        'departure0'    :'2020-07-01',
        'departure1'    :'2020-09-01',
        'arrival0'      :'2020-11-01',
        'arrival1'      :'2022-01-24',
        'mu'            : pd.sun['mu'],
        'step'          :1/sec2day,
        'frame'         : 'ECLIPJ2000',
        'observer'      :'SOLAR SYSTEM BARYCENTER',
        'cutoff_v'      :20.0,
        'c3_levels'     :None,
        'vinf_levels'   :None,
        'tof_levels'    :None,
        'dv_levels'     :None,
        'dv_cmap'       :'RdPu_r',
        'fig_size'      :(20,10),
        'lw'            :1.5,
        'title'         :'Pork chop plot',
        'fontsize'       :15,
        'show'          :False,
        'filename'      :None,
        'filename_dv'   :None,
        'dpi'           :300,
        'load'          :False
    }
    for key in config.keys():
        _config[key]= config[key]

    cutoff_c3 = config['cutoff v']**2

    #arrays of departure and arrival times
    et_departurs=np.arrange(
        spice.utc2et(_config['departure0']),
        spice.utc2et(_config['departure1'])+_config['step'],
        _config['step'])
    et_arrivals = np.arrange(
        spice.utc2et(_config['arrival0']),
        spice.utc2et(_config['arrival1'])+_config['step'],
        _config['step'])

    #number of days in each array and total combinations
    ds = len(et_departurs)
    as_ = len(et_arrivals)
    total = ds*as_

    print('Departure days : %i.'         % ds   )
    print('Arrival days : %i.'           % as_  )
    print('Total combinations: %i.'      % total)

    #create empty array for C3,v,infinity,and tof
    C3_shorts       = np.zeros((as_,ds))
    C3_longs        = np.zeros((as_, ds))
    v_inf_shorts    = np.zeros((as_, ds))
    v_inf_longs     = np.zeros((as_, ds))
    tofs            = np.zeros((as_,ds))

    #create arrays for indexing and meshgrid
    x = np.arrange(ds)
    y = np.arrange(as_)

    #for each combination
    for na in y:
        for nd in x:

            #state of planet0 at departure
            state_depart = st.calc_ephemeris(
                _config['plante0'],
                [et_departurs[ nd ] ],
                _config['frame' ],_config['observer'])[0]

            # state of planet1 at arrival
            state_arrival = st.calc_ephemeris(
                _config['plante1'],
                [et_arrivals[na]],
                _config['frame'], _config['observer'])[0]

            #caluclate flight time
            tof=et_arrivals[na]-et_departurs[nd]

            try:
                v_sc_depart_short,v_sc_arrive_short = lt.lamberts_universal_variables(
                    state_depart[:3],state_arrival[:3],
                    tof,tm=1,mu=_config['mu' ])
            except:
                v_sc_depart_short = np.array([1000,1000,1000])
                v_sc_arrive_short = np.array([1000,1000,1000])
            try:
                v_sc_depart_long, v_sc_arrive_long = lt.lamberts_universal_variables(
                    state_depart[:3], state_arrival[:3],
                    tof, tm=-1, mu=_config['mu'])
            except:
                v_sc_depart_long= np.array([1000,1000,1000])
                v_sc_arrive_short = np.array([1000, 1000, 1000])

                #calculate C3 values departing
                C3_short = nt.norm(v_sc_depart_short - state_depart[3:])**2
                C3_long  = nt.norm(v_sc_depart_long  - state_depart[3:])**2

                #check for unreasonable values
                if C3_short > cutoff_c3: C3_short=cutoff_c3
                if C3_long  > cutoff_c3: C3_long = cutoff_c3

                #calculate v infinity values arriving
                v_inf_short = nt.norm(v_sc_arrive_short - state_arrival[3:])
                v_inf_long  = nt.norm(v_sc_arrive_long  - state_arrival[3:])

                #check for unreasonable values
                if v_inf_short > _config['cutoff_v']:v_inf_short =_config['cutoff_v']
                if v_inf_long  > _config['cutoff_v']: v_inf_long  = _config['cutoff_v']

                #add values to corresponding arrays
                C3_shorts   [na,nd] = C3_short
                C3_longs    [na,nd] = C3_long
                v_inf_shorts[na,nd] = v_inf_short
                v_inf_longs [na,nd] = v_inf_long
                tofs        [na,nd] = tof

            print('%i / %i. '%(na,as_))

        tofs/=(3600.0*24.0

        #total delta v
        dv_shorts = v_inf_shorts + np.sqrt(C3_shorts )
        dv_longs  = v_inf_longs  + np.sqrt(C3_longs  )

        #create levels arrays
        if _config['c3_levels '   ] is None:
            _config['c3_levels'   ] = np.arrange( 10 , 50 , 2  )
        if _config['vinf_levels ' ] is None:
            _config['vinf_levels' ] = np.arrange( 0 , 15 , 1  )
        if _config['tof_levels '  ] is None:
            _config['tof_levels'  ] = np.arrange( 100 , 500 , 20  )
        if _config['dv_levels '   ] is None:
            _config['dv_levels'   ] = np.arrange( 3 , 20 , 0.5  )

        lw=_config[ 'lw' ]

        fig,ax=plt.subplots(figsize = config[ 'figsize' ])
        c0 = ax.contour(C3_shorts,
                levels = _config['c3_levels' ],     colors = 'm', linewidths = lw )
        c1 = ax.contour(C3_longs,
                levels=_config['c3_levels'   ], colors='m' , linewidths = lw )
        c2 = ax.contour(v_inf_shorts),
                        levels = _config['vinf_levels'], colors='deepskyblue', linewidths=lw)
        c3 = ax.contour(v_inf_longs),
                        levels = _config['vinf_levels'], colors = 'deepskyblue', linewidths = lw)
        c4 = ax.contour(tofs),
                        levels = _config['tof_levels' ], colors = 'white', linewidths = lw*0.6 )

        plt.clabel(c0,fmt= '%i')
        plt.clabel(c1, fmt='%i')
        plt.clabel(c2, fmt='%i')
        plt.clabel(c3, fmt='%i')
        plt.clabel(c4, fmt='%i')
        plt.plot([0], [0], 'm')
        plt.plot([0], [0], 'c')
        plt.plot([0], [0], 'w')
        plt.legend(
            [r'C3 ($\dfrac{km^2}{s^2}$)',
            r'$V_{\infty}\; (\dfrac{km}{s})$',
            r'Time of Flight (days)'],
            bbox_to_anchor=(1.005,1.01),
            fontsize = 10 )
        ax.set_title(_config['title']),fontsize=_config[ 'fontsize' ])
        ax.set_ylabel('Arrival (Days Past %s)' % _config['arrival0'], fontsize=_config['fontsize')
        ax.set_xlabel('"Departure (Days Past %s )' % _config['departure0'], fontsize=_config['fontsize')

        if _config['show']:
            plt.show()

        if _config['filename'] isnot None:

            plt.savefig(_config['filename']),dpi=_config['dpi'])
            print('Saved',_config['filename'])

        plt.close()


        #delta v plot


        fig,acx = plt.subplots(figsize=_config['figsize'])
        c0 = ax.contour(
            dv_shorts,levels=_config['dv_levels'],
            cmap = _config['dv_cmap'],linewidths=lw)
        c1 = ax.contour(
            dv_longs,levels=_config['dv_levels'],
            cmap = _config['dv_cmap'],linewidths=lw)
        c2 = ax.contour(tofs,
            levels = _config['tof_levels'],colors='c',linewidths=lw*0.6)
        plt.clabel(c0,fmt='%.1f')
        plt.clabel(c1, fmt='%.1f')
        plt.clabel(c2, fmt='%i')

        ax.set_title(_config['title_dv'],fontsize = _config['fontsize'])
        ax.set_ylabel('Arrival (Days past %s)' % _config['arrival0'], fontsize=_config['fontsize'])
        ax.set_xlabel('Departure (Days Past %s )' % _config['departure0'], fontsize=_config['fontsize'])

        if _config['show']:
            plt.show()

        if _config['filename_dv'] is not None:
            plt.savefig(_config['filename_dv'],dpi=_config['dpi'])
            print('Saved',_config['filename_dv'])
        plt.close()

    test=0


if __name__ == '__main__':
    interplanetry_porkchip(config=None)
