
package jsat.linear;

/**
 *
 * @author Edward Raff
 */
public interface Vec
{
    public int length();

    public double get(int index);

    public void set(int index, double val);
    
    public Vec add(double c);
    public Vec add(Vec b);
    public Vec subtract(double c);
    public Vec subtract(Vec b);
    public Vec multiply(double c);
    public Vec divide(double c);

    public Vec sortedCopy();

    public double min();
    public double max();
    public double sum();
    public double mean();
    public double standardDeviation();
    public double variance();
    public double median();
    public double skewness();
    public double kurtosis();
    
    public double dot(Vec v);



}
