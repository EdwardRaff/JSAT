package jsat.driftdetectors;

import java.util.*;
import jsat.math.OnLineStatistics;

/**
 * <i>Ad</i>aptive <i>Win</i>dowing (ADWIN) is an algorithm for detecting 
 * changes in an input stream. ADWIN maintains an approximated window of the 
 * input history, and works in O(log(n)) time and O(log(n)) memory, where 
 * <i>n</i> is the current window size. Whenever a drift is detected and 
 * handled, the size of the window will be reduced. <br>
 * <br>
 * The window in ADWIN is only for the {@code double} values passed when calling 
 * {@link #addSample(double, java.lang.Object) }. The object paired with the 
 * numeric value will <i>not</i> be compressed and is added to on every update. 
 * It is important to control its size using {@link #setMaxHistory(int) } when
 * using ADWIN. By default, ADWIN will use a maximum history of 0. <br>
 * <br>
 * See: Bifet, A.,&amp;Gavalda, R. (2007). <i>Learning from Time-Changing Data 
 * with Adaptive Windowing</i>. In SIAM International Conference on Data Mining.
 * 
 * @author Edward Raff
 */
public class ADWIN<V> extends BaseDriftDetector<V>
{

	private static final long serialVersionUID = 3287510845017257629L;
	private double delta;
    private OnLineStatistics allStats;
    /**
     * List of all the stats summarizing the windows. New items are added to the
     * tail end of the list so that we can iterate from the "head" down. This 
     * makes writing the logic somewhat easier. <br>
     * This means the head contains the oldest /largest items, and the tail 
     * contains the smallest / newest items. This is opposite of the ADWIN paper
     */
    private LinkedList<OnLineStatistics> windows;
    /*
     * default: "We use, somewhat arbitrarily, M = 5 for all experiments" under
     * section: 4 Experimental Validation of ADWIN2
     */
    private int M = 5;
    
    //Data used when a change is deteceted 
    private double leftMean = Double.NaN, leftVariance = Double.NaN;
    private double rightMean = Double.NaN, rightVariance = Double.NaN;


    /**
     * Creates a new ADWIN object for detecting changes in the mean value of a 
     * stream of inputs. It will use a not keep any object history by default. 
     * @param delta the desired false positive rate
     */
    public ADWIN(double delta)
    {
        this(delta, 0);
    }
    
    /**
     * Creates a new ADWIN object for detecting changes in the mean value of a 
     * stream of inputs. 
     * @param delta the desired false positive rate
     * @param maxHistory the maximum history of objects to keep
     */
    public ADWIN(double delta, int maxHistory)
    {
        super();
        setDelta(delta);
        setMaxHistory(maxHistory);
        allStats = new OnLineStatistics();
        windows = new LinkedList<OnLineStatistics>();
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public ADWIN(ADWIN<V> toCopy)
    {
        super(toCopy);
        this.delta = toCopy.delta;
        this.allStats = toCopy.allStats.clone();
        this.M = toCopy.M;
        this.leftMean = toCopy.leftMean;
        this.rightMean = toCopy.rightMean;
        this.leftVariance = toCopy.leftVariance;
        this.rightVariance = toCopy.rightVariance;
        this.windows = new LinkedList<OnLineStatistics>();
        for(OnLineStatistics stats : toCopy.windows)
            this.windows.add(stats.clone());
    }

    /**
     * Sets the upper bound on the false positive rate for detecting concept 
     * drifts
     * @param delta the upper bound on false positives in (0,1)
     */
    public void setDelta(double delta)
    {
        if(delta <= 0 || delta >= 1 || Double.isNaN(delta))
            throw new IllegalArgumentException("delta must be in (0,1), not " + delta);
        this.delta = delta;
    }

    /**
     * Returns the upper bound on false positives
     * @return the upper bound on false positives
     */
    public double getDelta()
    {
        return delta;
    }
    
    /**
     * This parameter controls the trade off of space and accuracy for the 
     * sliding window. The larger {@code M} becomes, the more accurate the 
     * window will be - but at the cost of execution speed. 
     * @param M the window space constant in [1, &infin;)
     */
    public void setM(int M)
    {
        if(M < 1)
            throw new IllegalArgumentException("M must be positive, not " + M);
        this.M = M;
    }
    
    /**
     * Returns the accuracy / speed parameter for ADWIN
     * @return the accuracy / speed parameter
     */
    public int getM()
    {
        return M;
    }
    
    @Override
    public boolean addSample(double value, V obj)
    {
        if(drifting)
            throw new UnhandledDriftException("Drift must be handled before continuing");
        time++;
        addToHistory(obj);
        //add to the window
        allStats.add(value);
        OnLineStatistics w = new OnLineStatistics();
        
        w.add(value);
        windows.addFirst(w);
        //check if a change has occured
        Iterator<OnLineStatistics> testIter = windows.descendingIterator();
        OnLineStatistics leftStats = new OnLineStatistics();
        OnLineStatistics rightStats = allStats.clone();
        
        final double deltaPrime  = delta/Math.log(allStats.getSumOfWeights());//will be > 1 in log, no issues
        final double ln2delta = Math.log(2) - Math.log(deltaPrime);
        final double variance_W = allStats.getVarance();
        while(testIter.hasNext())
        {
            OnLineStatistics windowItem = testIter.next();
            
            //accumulate left side statistics
            leftStats.add(windowItem);
            //decrament right side stats
            rightStats.remove(windowItem);
            
            double n_0 = leftStats.getSumOfWeights();
            double n_1 = rightStats.getSumOfWeights();
            double mu_0 = leftStats.getMean();
            double mu_1 = rightStats.getMean();
            //  1/(1/x+1/y) = x y / (x + y), and then inverse so (x+y)/(xy)
            double mInv = (n_0 + n_1) / (n_0 * n_1);

            double e_cut = Math.sqrt(2 * mInv * variance_W * ln2delta) + 2.0 / 3.0 * mInv * ln2delta;

            if(Math.abs(mu_0 - mu_1) > e_cut)//CHANGE! OMG
            {
                drifting = true;
                driftStart = (int) (n_0);
                //set stats for them to find
                leftMean = mu_0;
                leftVariance = leftStats.getVarance();
                rightMean = mu_1;
                rightVariance = rightStats.getVarance();
                /*
                 * we keep going incase there is a more recent start point for 
                 * the drift, as the change in mean at the front could have been
                 * large enough / dramatic enough to make preciding cuts also 
                 * look like drift
                 */
            }
        }
        compress();
        
        return drifting;
    }
    
    /**
     * Compresses the current window
     */
    private void compress()
    {
        //compress
        ListIterator<OnLineStatistics> listIter = windows.listIterator();
        double lastSizeSeen = -Double.MAX_VALUE;
        int lastSizeCount = 0;
        
        while(listIter.hasNext())
        {
            OnLineStatistics window = listIter.next();
            double n = window.getSumOfWeights();
            if(n == lastSizeSeen)
            {
                if(++lastSizeCount > M)//compress, can only occur if there is a previous
                {
                    listIter.previous();
                    window.add(listIter.previous());
                    listIter.remove();//remove the preivous
                    if(listIter.hasNext())
                        listIter.next();//back to where we were, which has been modified
                    //so nowe we must be looking at a new range since we just promoted a window
                    lastSizeSeen = window.getSumOfWeights();
                    lastSizeCount = 1;
                }
            }
            else
            {
                lastSizeSeen = n;
                lastSizeCount = 1;
            }
        }
    }
    
    /**
     * Returns the mean value for all inputs contained in the current window
     * @return the mean value of the window
     */
    public double getMean()
    {
        return allStats.getMean();
    }
    
    /**
     * Returns the variance for all inputs contained in the current window
     * @return the variance for the window
     */
    public double getVariance()
    {
        return allStats.getVarance();
    }
    
    /**
     * Returns the standard deviation for all inputs contained in the current 
     * window. 
     * @return the standard deviation for the window
     */
    public double getStndDev()
    {
        return allStats.getStandardDeviation();
    }
    
    /**
     * This returns the current "length" of the window, which is the number of 
     * items that have been added to the ADWIN object since the last drift, and 
     * is reduced when drift occurres. 
     * @return the number of items stored implicitly in the window
     */
    public int getWidnowLength()
    {
        return time;
    }
    
    /**
     * Returns the mean value determined for the older values that we have 
     * drifted away from. <br>
     * If drifting has not occurred or has already been handled, 
     * {@link Double#NaN} will be returned.
     * @return the mean for the old values. 
     */
    public double getOldMean()
    {
        return leftMean;
    }
    
    /**
     * Returns the variance for the older values that we have 
     * drifted away from. <br>
     * If drifting has not occurred or has already been handled, 
     * {@link Double#NaN} will be returned.
     * @return the variance for the old values
     */
    public double getOldVariance()
    {
        return leftVariance;
    }
    
    /**
     * Returns the standard deviation for the older values that we have 
     * drifted away from. <br>
     * If drifting has not occurred or has already been handled, 
     * {@link Double#NaN} will be returned.
     * @return the standard deviation for the old values
     */
    public double getOldStndDev()
    {
        return Math.sqrt(leftVariance);
    }
    
    /**
     * Returns the mean value determined for the newer values that we have 
     * drifted into. <br>
     * If drifting has not occurred or has already been handled, 
     * {@link Double#NaN} will be returned.
     * @return the mean for the newer values. 
     */
    public double getNewMean()
    {
        return rightMean;
    }
    
    /**
     * Returns the variance for the newer values that we have 
     * drifted into. <br>
     * If drifting has not occurred or has already been handled, 
     * {@link Double#NaN} will be returned.
     * @return the variance for the newer values
     */
    public double getNewVariance()
    {
        return rightVariance;
    }
    
    /**
     * Returns the standard deviation for the newer values that we have 
     * drifted into. <br>
     * If drifting has not occurred or has already been handled, 
     * {@link Double#NaN} will be returned.
     * @return the standard deviation for the newer values
     */
    public double getNewStndDev()
    {
        return Math.sqrt(rightVariance);
    }

    /**
     * This implementation of ADWIN allows for choosing to drop either the old 
     * values, as is normal for a drift detector, <i>or</i> to drop the newer 
     * values. Passing {@code true} will result in the standard behavior of 
     * calling {@link #driftHandled() }. <br>
     * If {@code false} is passed in to drop the <i>newer</i> values that 
     * drifted it is probably that continuing to add new examples will continue 
     * to cause detections. 
     * 
     * @param dropOld {@code true} to drop the older values out of the window 
     * that we drifted away from, or {@code false} to drop the newer values and
     * retain the old ones. 
     */
    public void driftHandled(boolean dropOld)
    {
        /*
         * Iterate through and either drop everything to the left OR the right
         * Track statiscits so that we can update allStats
         */
        Iterator<OnLineStatistics> testIter = windows.descendingIterator();
        OnLineStatistics leftStats = new OnLineStatistics();
     
        while (testIter.hasNext())
        {
            OnLineStatistics windowItem = testIter.next();

            //accumulate left side statistics
            if(leftStats.getSumOfWeights() < driftStart)
            {
                leftStats.add(windowItem);
                if(dropOld)
                    testIter.remove();
            }
            else
            {
                if(!dropOld)
                    testIter.remove();
            }
        }
        
        if(dropOld)
            allStats.remove(leftStats);
        else
            allStats = leftStats;
        time = (int) allStats.getSumOfWeights();

        leftMean = leftVariance = rightMean = rightVariance = Double.NaN;

        //Calling at the end b/c we need driftStart's value
        super.driftHandled();
    }

    @Override
    public void driftHandled()
    {
        this.driftHandled(true);
    }
    
    @Override
    public ADWIN<V> clone()
    {
        return new ADWIN<V>(this);
    }
    
}
