
package jsat.classifiers;

import java.io.Serializable;
import jsat.linear.Vec;

/**
 *
 * This class exists so that any data point can be arbitrarily paired with some value
 * @author Edward Raff
 */
public class DataPointPair<P> implements Serializable
{

	private static final long serialVersionUID = 5091308998873225566L;
	DataPoint dataPoint;
    P pair;

    public DataPointPair(DataPoint dataPoint, P pair)
    {
        this.dataPoint = dataPoint;
        this.pair = pair;
    }

    public void setDataPoint(DataPoint dataPoint)
    {
        this.dataPoint = dataPoint;
    }

    public void setPair(P pair)
    {
        this.pair = pair;
    }

    public DataPoint getDataPoint()
    {
        return dataPoint;
    }

    public P getPair()
    {
        return pair;
    }
    
    /**
     * The same as calling {@link DataPoint#getNumericalValues() } on {@link #getDataPoint() }.
     * @return the Vec related to the data point in this pair. 
     */
    public Vec getVector()
    {
        return dataPoint.getNumericalValues();
    }
}
