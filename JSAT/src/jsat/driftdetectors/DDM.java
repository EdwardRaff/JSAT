package jsat.driftdetectors;

/**
 * DDM (Drift Detection Method) is a drift detector for binary events, and is 
 * meant to detect decreases in the success rate over time. As such it will not
 * inform of any positive drift. <br>
 * <br>
 * This drift detector supports a warning state, and will not begin to store 
 * the object history until a warning state begins. If the warning state ends 
 * before a detection of drift occurs, the history will be dropped. 
 * <br>
 * See: Gama, J., Medas, P., Castillo, G.,&amp;Rodrigues, P. (2004). <i>Learning 
 * with Drift Detection</i>. In A. C. Bazzan&amp;S. Labidi (Eds.), Advances in 
 * Artificial Intelligence – SBIA 2004 (pp. 286–295). Springer Berlin 
 * Heidelberg. doi:10.1007/978-3-540-28645-5_29
 * 
 * @author Edward Raff
 */
public class DDM<V> extends BaseDriftDetector<V>
{

	private static final long serialVersionUID = 3023405445609636195L;

	/**
     * Number of times we won the trial
     */
    private int fails;

    private int minSamples = 30;
    
    private double p_min;
    private double s_min;
    
    private double warningThreshold;
    private double driftThreshold;

    /**
     * Creates a new DDM drift detector using the default warning and drift 
     * thresholds of 2 and 3 respectively. 
     */
    public DDM()
    {
        this(2, 3);
    }
    
    /**
     * Creates a new DDM drift detector
     * @param warningThreshold the  threshold for starting a warning state
     * @param driftThreshold the threshold for recognizing a drift
     */
    public DDM(double warningThreshold, double driftThreshold)
    {
        super();
        setWarningThreshold(warningThreshold);
        setDriftThreshold(driftThreshold);
        driftHandled();
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public DDM(DDM<V> toCopy)
    {
        super(toCopy);
        this.fails = toCopy.fails;
        this.warningThreshold = toCopy.warningThreshold;
        this.driftThreshold = toCopy.driftThreshold;
    }
    
    /**
     * Returns the current estimate of the success rate (number of {@code true}
     * inputs) for the model. 
     * @return the current estimate of the success rate
     */
    public double getSuccessRate()
    {
        return 1.0-fails/(double)time;
    }
    
    /**
     * Adds a new boolean trial to the detector, with the goal of detecting when
     * the number of successful trials ({@code true}) drifts to a new value. 
     * This detector begins storing a history of the {@code obj} inputs only
     * once it has entered a warning state. <br>
     * This detector is specifically meant to detect drops in the success rate, 
     * and will not cause any warning or drift detections for increases in the 
     * success rate. 
     * @param trial the result of the trial
     * @param obj the object to associate with the trial
     * @return {@code true} if we are in a warning or drift state, 
     * {@code false } if we are not
     */
    public boolean addSample(boolean trial, V obj)
    {
        if(drifting)
            throw new UnhandledDriftException();
        if(!trial)
            fails++;
        time++;
        
        if(time < minSamples)
            return false;
        
        final double p_i = fails/(double)time;
        final double s_i = Math.sqrt(p_i*(1-p_i)/time);
        final double ps = p_i+s_i;
        
        //values are updated when pi +si is lower than pmin +smin
        if(ps < p_min + s_min)
        {
            p_min = p_i;
            s_min = s_i;
        }
        
        if (ps > p_min + warningThreshold * s_min)
        {
            if(!warning)//first entry 
            {
                warning = true;
                driftStart = time - 1;
            }
            addToHistory(obj);
            if (ps > p_min + driftThreshold * s_min)
            {
                warning = false;
                drifting = true;
            }
            return true;
        }
        else//everything is good
        {
            warning = false;
            driftStart = -1;
            clearHistory();
            return false;
        }
    }

    /**
     * Sets the multiplier on the standard deviation that must be exceeded to 
     * initiate a warning state. Once in the warning state, DDM will begin to 
     * collect a history of the inputs <br>
     * Increasing the warning threshold makes it take longer to start detecting 
     * a change, but reduces false positives. <br>
     * If the warning threshold is set above the 
     * {@link #setDriftThreshold(double) }, the drift state will not occur until
     * the warning state is reached, and the warning state will be skipped. 
     * @param warningThreshold the positive multiplier threshold for starting a 
     * warning state
     */
    public void setWarningThreshold(double warningThreshold)
    {
        if(warningThreshold <= 0 || Double.isNaN(warningThreshold) || Double.isInfinite(warningThreshold))
            throw new IllegalArgumentException("warning threshold must be positive, not " + warningThreshold);
        this.warningThreshold = warningThreshold;
    }

    /**
     * Returns the threshold multiple for controlling the false positive / 
     * negative rate on detecting changes. 
     * @return the threshold multiple for controlling warning detection
     */
    public double getWarningThreshold()
    {
        return warningThreshold;
    }

    /**
     * Sets the multiplier on the standard deviation that must be exceeded to 
     * recognize the change as a drift. <br>
     * Increasing the drift threshold makes it take longer to start detecting 
     * a change, but reduces false positives.
     * @param driftThreshold the positive multiplier threshold for detecting a
     * drift
     */
    public void setDriftThreshold(double driftThreshold)
    {
        if(driftThreshold <= 0 || Double.isNaN(driftThreshold) || Double.isInfinite(driftThreshold))
            throw new IllegalArgumentException("Dritf threshold must be positive, not " + driftThreshold);
        this.driftThreshold = driftThreshold;
    }

    /**
     * Returns the threshold multiple for controlling the false positive / 
     * negative rate on detecting changes.
     * @return the threshold for controlling drift detection
     */
    public double getDriftThreshold()
    {
        return driftThreshold;
    }

    @Override
    public boolean addSample(double value, V obj)
    {
        return addSample(value == 0.0, obj);
    }

    @Override
    public void driftHandled()
    {
        super.driftHandled();
        fails = 0;
        p_min = s_min = Double.POSITIVE_INFINITY;
        time = 0;
        clearHistory();
    }

    @Override
    public DDM<V> clone()
    {
        return new DDM<V>(this);
    }
    
}
