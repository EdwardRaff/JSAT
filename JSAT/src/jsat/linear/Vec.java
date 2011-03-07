
package jsat.linear;

/**
 *
 * @author Edward Raff
 */
public interface Vec<V extends Vec>
{
    public int length();

    public double get(int index);

    public void set(int index, double val);
    
    public V add(V b);
    public V subtract(V b);
    public V multiply(V b);
    public V divide(V b);

    public V sortedCopy();

    public double min();
    public double max();
    public double sum();
    public double mean();
    public double standardDeviation();
    public double median();



}
