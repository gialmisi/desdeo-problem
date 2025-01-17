General Aviation Aircraft (GAA) product family design problem
==============================================================
With nine design variables per aircraft and three aircraft (2-seater, 4-seater and 6-seater) 
in the family, there are 27 design variables [table_1]_. There are 10 objectives [table_2]_ and
one constraint used in this formulation. The code is from https://github.com/matthewjwoodruff/generalaviation
and more details about the test problem can be found in [1]_, [2]_, [3]_ and [4]_.


.. [table_1]
  ===  ========================  =========  =========
  Design variables
  ---------------------------------------------------
  No.  Name                      Minimum    Maximum
  ===  ========================  =========  =========
   1    Nominal cruising speed      0.24     0.48
   2    Aspect ratio                7        11
   3    Wing sweep                  0        6
   4    Propeller diameter          5.5      5.968
   5    Wing loading                19       25
   6    Activity factor             85       110
   7    Seat width                  14       20
   8    Tail elonagation            3        3.75
   9    Wing taper                  0.46     1
  ===  ========================  =========  =========


.. [table_2]
  ===  =====================  =============================  ===========
  Objectives
  ----------------------------------------------------------------------
  No.  Objectives              Value                         Preference
  ===  =====================  =============================  ===========
   1    Maximum NOISE         Max (NOISE2, NOISE4 NOISE6)    Minimise
   2    Maximum WEMP          Max (WEMP2, WEMP4, WEMP6)      Minimise
   3    Maximum DOC           Max (DOC2, DOC4, DOC6)         Minimise
   4    Maximum ROUGH         Max (ROUGH2, ROUGH4, ROUGH6)   Minimise
   5    Maximum WFUEL         Max (WFUEL2, WFUEL4, WFUEL6)   Minimise
   6    Maximum PURCH         Max (PURCH2, PURCH4, PURCH6)   Minimise
   7    Minimum RANGE         Min (RANGE2, RANGE4, RANGE6)   Maximise
   8    Minimum max LDMAX     Min (LDMAX2, LDMAX4, LDMAX6)   Maximise
   9    Minimum max VCMAX     Min (VCMAX2, VCMAX4, VCMAX6)   Maximise
  10    PFPF                    -                            Minimise
  ===  =====================  =============================  ===========

.. [1] T. W. Simpson, W. Chen, J. K. Allen, and F. Mistree (1996), 
  "Conceptual design of a family of products through the use of the robust
  concept exploration method," in 6th AIAA/USAF/NASA/ ISSMO Symposium on 
  Multidiciplinary Analysis and Optimization, vol. 2, pp. 1535-1545.

.. [2] T. W. Simpson, B. S. D'Souza (2004), "Assessing variable levels of platform 
  commonality within a product family using a multiobjective genetic algorithm," 
  Concurrent Engineering: Research and Applications, vol. 12, no. 2, pp. 119-130.

.. [3] R. Shah, P. M. Reed, and T. W. Simpson (2011), "Many-objective evolutionary optimization 
  and visual analytics for product family design," Multiobjective Evolutionary Optimisation 
  for Product Design and Manufacturing, Springer, London, pp. 137-159.

.. [4] M. Woodruff, T. W. Simpson, P. M. Reed (2013), "Diagnostic Analysis of Metamodels' 
  Multivariate Dependencies and their Impacts in Many-Objective Design Optimization," 
  Proceedings of the ASME 2013 IDETC/CIE Conference, Paper No. DETC2013-13125.