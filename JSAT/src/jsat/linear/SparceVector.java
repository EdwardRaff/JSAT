
package jsat.linear;

import java.util.Arrays;
import static java.lang.Math.*;

/**
 *
 * @author Edward Raff
 */
public class SparceVector implements Vec
{
    /**
     * Length of the vector
     */
    private int length;
    /**
     * number of indices used in this vector
     */
    private int used;
    /**
     * The mapping to true index values
     */
    private int[] indexes;
    /**
     * The Corresponding values for each index
     */
    private double[] values;
    
    private Double sumCache = null;
    private Double varianceCache = null;
    private Double minCache = null;
    private Double maxCache = null;
    
    public SparceVector(int length)
    {
        this(length, 10);
    }

    
    
    public SparceVector(int length, int capacity)
    {
        this.length = length;
        this.indexes = new int[capacity];
        this.values = new double[capacity];
    }
    
    /**
     * nulls out the cached summary statistics, should be called every time the data set changes
     */
    private void clearCaches()
    {
        sumCache = null;
        varianceCache = null;
        minCache = null;
        maxCache = null;
    }
    
    public int length()
    {
        return length;
    }

    /**
     * @return the number of non zero elements in the vector.
     */
    public int nnz()
    {
        return used;
    }
    
    public double get(int index)
    {
        if (index > length - 1 || index < 0)
            throw new ArithmeticException("Can not access an index larger then the vector or a negative index");

        int location = Arrays.binarySearch(indexes, 0, used, index);

        if (location < 0)
            return 0.0;
        else
            return values[location];
    }

    public void set(int index, double val)
    {
        if(index > length-1 || index < 0)
            throw new ArithmeticException("Can not set an index larger then the array");

        
        clearCaches();
        int insertLocation = Arrays.binarySearch(indexes, 0, used, index);
        if(insertLocation >= 0)
            values[insertLocation] = val;
        else//More complicated
        {
            insertLocation = -(insertLocation+1);//Convert from negative value to the location is should be placed, see JavaDoc of binarySearch
            if(used == indexes.length)//Full, expand
            {
                int newIndexesSize = indexes.length*3/2;
                indexes = Arrays.copyOf(indexes, newIndexesSize);
                values = Arrays.copyOf(values, newIndexesSize);
            }

            if(insertLocation < used)//Instead of moving indexes over manualy, set it up to use a native System call to move things out of the way
            {
                System.arraycopy(indexes, insertLocation, indexes, insertLocation+1, used-insertLocation);
                System.arraycopy(values, insertLocation, values, insertLocation+1, used-insertLocation);
            }

            indexes[insertLocation] = index;
            values[insertLocation] = val;
            used++;
        }
    }

    public Vec add(Vec v)
    {
        if(this.length() != v.length())
            throw new ArithmeticException("Vectors must have the same length");

        
        if(v instanceof SparceVector)
        {
            SparceVector ret = new SparceVector(length());
            SparceVector b = (SparceVector) v;
            int p1 = 0, p2 = 0;
            while (p1 < used && p2 < b.used)
            {
                int a1 = indexes[p1], a2 = b.indexes[p2];
                if (a1 == a2)
                {
                    ret.set(a1, values[p1] + b.values[p2]);
                    p1++;
                    p2++;
                }
                else if (a1 > a2)
                {
                    ret.set(a2, b.values[p2]);
                    p2++;
                }
                else
                {
                    ret.set(a1, values[p1]);
                    p1++;
                }
            }
            
            return ret;
        }
        
        //Else we are sparce, and they are dense
        DenseVector ret = ((DenseVector) v).deepCopy();
        int p1 = 0;
        for(int i = 0; i < used; i++)
            ret.set(indexes[i], ret.get(indexes[i]) + values[i]); 
        
        return ret;
    }

    public Vec subtract(Vec v)
    {
        if(this.length() != v.length())
            throw new ArithmeticException("Vectors must have the same length");

        
        if(v instanceof SparceVector)
        {
            SparceVector ret = new SparceVector(length());
            SparceVector b = (SparceVector) v;
            int p1 = 0, p2 = 0;
            while (p1 < used && p2 < b.used)
            {
                int a1 = indexes[p1], a2 = b.indexes[p2];
                if (a1 == a2)
                {
                    ret.set(a1, values[p1] - b.values[p2]);
                    p1++;
                    p2++;
                }
                else if (a1 > a2)
                {
                    ret.set(a2, -b.values[p2]);
                    p2++;
                }
                else
                {
                    ret.set(a1, values[p1]);
                    p1++;
                }
            }
            
            return ret;
        }
        
        //Else we are sparce, and they are dense
        DenseVector ret = ((DenseVector) v).deepCopy();
        for(int i = 0; i < used; i++)
            ret.set(indexes[i], ret.get(indexes[i]) - values[i]); 
        
        return ret;
    }


    public Vec sortedCopy()
    {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    public double min()
    {
        if(minCache != null)
            return minCache;
        double result = 0;
        for(int i = 0; i < used; i++)
            result = Math.min(result, values[i]);

        return (minCache = result);
    }

    public double max()
    {
        if(maxCache != null)
            return maxCache;
        
        double result = 0;
        for(int i = 0; i < used; i++)
            result = Math.max(result, values[i]);

        return (maxCache = result);
    }

    public double sum()
    {
        if(sumCache != null)
            return sumCache;
        
        /*
         * Uses Kahan summation algorithm, which is more accurate then
         * naively summing the values in floating point. Though it
         * does not guarenty the best possible accuracy
         *
         * See: http://en.wikipedia.org/wiki/Kahan_summation_algorithm
         */

        double sum = 0;
        double c = 0;
        for(double d : values)
        {
            double y = d - c;
            double t = sum+y;
            c = (t - sum) - y;
            sum = t;
        }

        return (sumCache = sum);
    }

    public double mean()
    {
        return sum()/length();
    }

    public double standardDeviation()
    {
        return Math.sqrt(variance());
    }

    public double variance()
    {
        if(varianceCache != null)
            return varianceCache;
        
        double mu = mean();
        double tmp = 0;

        double N = length();


        for(double x : values)
            tmp += Math.pow(x-mu, 2)/N;
        //Now add all the zeros into it
        tmp +=  (length()-used) * Math.pow(0-mu, 2)/N;
        
        return (varianceCache = tmp);
    }

    public double median()
    {
        throw new UnsupportedOperationException("Not supported yet.");
    }
    
    public double skewness()
    {
        double mean = mean();
        
        double tmp = 0;
        
        for(int i = 0; i < used; i++)
            tmp += pow(values[i]-mean, 3);
        
        //All the zero's we arent storing
        tmp += pow(-mean, 3)*(length-used);
        
        double s1 = tmp / (pow(standardDeviation(), 3) * (length-1) );
        
        if(length >= 3)//We can use the bias corrected formula
            return sqrt(length*(length-1))/(length-2)*s1;
        
        return s1;
    }

    public double kurtosis()
    {
        double mean = mean();
        
        double tmp = 0;
        
        for(int i = 0; i < used; i++)
            tmp += pow(values[i]-mean, 4);
        
        //All the zero's we arent storing
        tmp += pow(-mean, 4)*(length-used);
        
        return tmp / (pow(standardDeviation(), 4) * (length-1) ) - 3;
    }

    public double dot(Vec v)
    {
        double dot = 0;

        if(v instanceof SparceVector)
        {
            SparceVector b = (SparceVector) v;
            int p1 = 0, p2 = 0;
            while (p1 < used && p2 < b.used)
            {
                int a1 = indexes[p1], a2 = b.indexes[p2];
                if (a1 == a2)
                {
                    dot += values[p1] * b.values[p2];
                    p1++;
                    p2++;
                }
                else if (a1 > a2)
                    p2++;
                else
                    p1++;
            }
        }
        
        //Else it is dense
        
        for(int i = 0; i < length(); i++)
        {
            dot += values[i] * v.get(indexes[i]);
        }
        
        return dot;
    }

    @Override
    public String toString()
    {
        StringBuilder sb = new StringBuilder("[");
        
        int p = 0;
        for(int i = 0; i < length(); i++)
        {
            if(i != 0)
                sb.append(", ");
            
            if(p < used && indexes[p] == i)
                sb.append(values[p++]);
            else
                sb.append("0");
        }
        sb.append("]");
        
        return sb.toString();
    }

    public Vec multiply(double c)
    {
        SparceVector sv = new SparceVector(length, used);
        
        for(int i = 0; i < used; i++)
            sv.values[i] *= c;
        
        return sv;
    }

    public Vec divide(double c)
    {
        SparceVector sv = new SparceVector(length, used);
        
        for(int i = 0; i < used; i++)
            sv.values[i] /= c;
        
        return sv;
    }

    public Vec add(double c)
    {
        SparceVector sv = new SparceVector(length, used);
        
        for(int i = 0; i < used; i++)
            sv.values[i] += c;
        
        return sv;
    }

    public Vec subtract(double c)
    {
        SparceVector sv = new SparceVector(length, used);
        
        for(int i = 0; i < used; i++)
            sv.values[i] -= c;
        
        return sv;
    }
    
    
    
}
