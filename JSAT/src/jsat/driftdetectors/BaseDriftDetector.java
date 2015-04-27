package jsat.driftdetectors;

import java.io.Serializable;
import java.util.*;

/**
 * Base class for providing common functionality to drift detection algorithms
 * @author Edward Raff
 */
public abstract class BaseDriftDetector<V> implements Cloneable, Serializable
{

	private static final long serialVersionUID = -5857845807016446270L;

	/**
     * Tracks the number of updates / trial scene. May be reset as needed, so 
     * long as it increases compared to {@link #driftStart}
     */
    protected int time = 0;
    
    /**
     * Controls the maximum amount of history to keep
     */
    protected int maxHistory = Integer.MAX_VALUE;
    
    /**
     * Set to {@code true} to indicate that a warning mode in in effect. 
     */
    protected boolean warning = false;
    /**
     * Set to {@code true} to indicate that concept drift has occurred 
     */
    protected boolean drifting = false;
    
    /**
     * Set this value to the time point where the drift is believed to have 
     * started from. Set to -1 to indicate no drift
     */
    protected int driftStart = -1;
    
    /**
     * Holds the associated object history. The history is always FIFO, with the
     * end (tail) of the queue containing the most recent object, and the front 
     * (head) containing the oldest object. 
     */
    protected Deque<V> history;

    protected BaseDriftDetector()
    {
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    protected BaseDriftDetector(BaseDriftDetector<V> toCopy)
    {
        this.time = toCopy.time;
        this.maxHistory = toCopy.maxHistory;
        this.warning = toCopy.warning;
        this.driftStart = toCopy.driftStart;
        if(toCopy.history != null)
        {
            this.history = new ArrayDeque<V>(toCopy.history.size());
            for(V v : toCopy.history)
                this.history.add(v);
        }
    }
    
    /**
     * Returns {@code true} if the algorithm is in a warning state. This state 
     * indicates that the algorithm believes concept drift may be occurring, but
     * is not confident enough to say that is had definitely occurred. <br>
     * Not all algorithms will raise a warning state, and some may only begin 
     * keeping track of history once in the warning state.
     * @return {@code true} if concept drift may have started, but is not sure 
     */
    public boolean isWarning()
    {
        return warning;
    }
    
    /**
     * Returns {@code true} if the algorithm believes that drift has definitely 
     * occurred. At this 
     * @return 
     */
    public boolean isDrifting()
    {
        return drifting;
    }

    /**
     * Returns the maximum number of items that will be kept in the history. 
     * @return the maximum number of items that will be kept in the history. 
     */
    public int getMaxHistory()
    {
        return maxHistory;
    }

    /**
     * Sets the maximum number of items to store in history. Setting this to 
     * {@code 0} will keep the detector from ever storing history. <br>
     * The user can still keep their own independent history or checkpoints by 
     * using the {@link #isDrifting() } and {@link #isWarning() } methods. <br>
     * <br>
     * The history size may be changed at any time, but may result in the loss 
     * of history. 
     * 
     * @param maxHistory the new maximum history size of objects added
     */
    public void setMaxHistory(int maxHistory)
    {
        this.maxHistory = maxHistory;
        if(history != null)
            if (this.maxHistory == 0)
                history.clear();
            else
                while (history.size() > maxHistory)
                    history.removeFirst();
    }
    
    /**
     * Adds the given item to the history, creating a new history holder if 
     * needed. This method handles the cases where the max history is zero, 
     * and when the history is full (dropping the oldest)
     * @param obj the object to add to the history
     */
    protected void addToHistory(V obj)
    {
        if(maxHistory < 1)
            return;
        if(history == null)
            if (maxHistory != Integer.MAX_VALUE)//we probably set it to a reasonable value
            {
                try
                {
                    history = new ArrayDeque<V>(maxHistory);
                }
                catch (Exception ex)
                {
                    //what is we cause one of the many OOM exceptiosn b/c initial history was too big?
                    //AKA we googed on being helpful
                    history = new ArrayDeque<V>();
                }
            }
            else
                history = new ArrayDeque<V>();
        if(history.size() == maxHistory)//make room
            history.removeFirst();
        history.add(obj);
    }
   
    /**
     * Clears the current history
     */
    public void clearHistory()
    {
        if(history != null)
            history.clear();
    }
    
    /**
     * Returns the number of items in recent history that differed from the 
     * historical values, or {@code -1} if there has not been any detected 
     * drift. This method will return {@code -1} even if {@link #isWarning() } 
     * is {@code true}. 
     * @return the number of updates ago that the drift started, or {@code -1}
     * if no drift has occurred
     */
    public int getDriftAge()
    {
        if(driftStart == -1)
            return -1;
        return time-driftStart;
    }
    
    
    /**
     * Returns a new list containing up to {@link #getMaxHistory() } objects in
     * the history that drifted away from the prior state of the model. <br>
     * The 0 index in the list will be the most recently added item, and the 
     * largest index will be the oldest item. 
     * @return the list of objects that make up the effected history
     */
    public List<V> getDriftedHistory()
    {
        int historyToGram = Math.min(time - driftStart, history.size());
        ArrayList<V> histList = new ArrayList<V>(historyToGram);
        Iterator<V> histIter = history.descendingIterator();
        while(histIter.hasNext() && historyToGram > 0)
        {
            historyToGram--;
            histList.add(histIter.next());
        }
        return histList;
    }
    
    /**
     * Adds a new point to the drift detector. If an escalation in state occurs, 
     * {@code true} will be returned. A change of state could be either drift 
     * occurring {@link #isDrifting} or a warning state {@link #isWarning}. 
     * If the detector was in a warning state and then goes back to normal, 
     * {@code false} will be returned. <br>
     * <br>
     * For binary (true / false) drift detectors, {@code value} will be 
     * considered {@code false} if and only if its value is equal to zero. Any
     * non zero value will be treated as {@code true} <br>
     * <br>
     * Objects added with the value may or may not be added to the history, the
     * behavior is algorithm dependent. Some may always add it to the history, 
     * while others will only begin collecting history once a warning state 
     * occurs. 
     * 
     * @param value the numeric value to add to the drift detector
     * @param obj the object associated with this value. It may or may not be 
     * stored in the detectors history
     * @return {@code true} if a drift has or may be starting. 
     * @throws UnhandledDriftException if {@link #driftHandled() } is not called 
     * after drifting is detected
     */
    public abstract boolean addSample(double value, V obj);
    
    /**
     * This method should be called once the drift is handled. Once done, this 
     * method will clear the flags and prepare the detector to continue tracking 
     * drift again. <br>
     * By using this method, one can continue to track multiple future drift 
     * changes without having to feed the history data (which may be incomplete)
     * into a new detector object. 
     */
    public void driftHandled()
    {
        warning = drifting = false;
        driftStart = -1;
    }

    @Override
    abstract public Object clone();
    
            
}
