
package jsat.classifiers;

/**
 *
 * This class exists so that any data point can be arbitrarily paired with some value
 * @author Edward Raff
 */
public class DataPointPair<P>
{
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
}
