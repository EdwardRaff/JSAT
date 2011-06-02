
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
    
    public void mutableAdd(double c);
    public void mutableAdd(Vec b);
    public void mutableSubtract(double c);
    public void mutableSubtract(Vec b);
    public void mutableMultiply(double c);
    public void mutableDivide(double c);

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
    
    /**
     * 
     * @param v the other vector
     * @return  the dot product of this vector and another
     */
    public double dot(Vec v);

    @Override
    public String toString();


}
