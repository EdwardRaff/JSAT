
package jsat.linear;

import java.util.ArrayList;
import java.util.Collections;

import static java.lang.Math.*;

/**
 *
 * @author Edward Raff
 */
public class DenseVector implements Vec<DenseVector>
{
    private ArrayList<Double> array;

    public DenseVector()
    {
        array = new ArrayList<Double>();
    }

    public DenseVector(int initalSize)
    {
        array = new ArrayList<Double>(initalSize);
        for(int i = 0; i < initalSize; i++)
            array.add(0.0);
    }

    public DenseVector(ArrayList<Double> array)
    {
        this.array = array;
    }
    
    public int length()
    {
        return array.size();
    }

    public double get(int index)
    {
        return array.get(index);
    }

    public void set(int index, double val)
    {
        array.set(index, val);
    }

    public DenseVector add(DenseVector b)
    {
        if(b.length() != this.length())
            throw new ArithmeticException("Vectors are of uneven length");
        ArrayList<Double> result = new ArrayList<Double>(length());
        for(int i = 0; i < length(); i++)
            result.add(this.get(i)+b.get(i));

        return new DenseVector(result);
    }

    public DenseVector subtract(DenseVector b)
    {
        if(b.length() != this.length())
            throw new ArithmeticException("Vectors are of uneven length");
        ArrayList<Double> result = new ArrayList<Double>(length());
        for(int i = 0; i < length(); i++)
            result.add(this.get(i)-b.get(i));

        return new DenseVector(result);
    }

    public DenseVector multiply(DenseVector b)
    {
        if(b.length() != this.length())
            throw new ArithmeticException("Vectors are of uneven length");
        ArrayList<Double> result = new ArrayList<Double>(length());
        for(int i = 0; i < length(); i++)
            result.add(this.get(i)*b.get(i));

        return new DenseVector(result);
    }

    public DenseVector divide(DenseVector b)
    {
        if(b.length() != this.length())
            throw new ArithmeticException("Vectors are of uneven length");
        ArrayList<Double> result = new ArrayList<Double>(length());
        for(int i = 0; i < length(); i++)
            result.add(this.get(i)/b.get(i));
        
        return new DenseVector(result);
    }

    public double dot(DenseVector b)
    {
        if(b.length() != this.length())
            throw new ArithmeticException("Vectors are of uneven length");
        double sum = 0;

        for(int i = 0; i < length(); i++)
            sum += this.get(i)*b.get(i);

        return sum;
    }

    public double min()
    {
        double result = array.get(0);
        for(int i = 1; i < array.size(); i++)
            result = Math.min(result, array.get(i));

        return result;
    }

    public double max()
    {
        double result = array.get(0);
        for(int i = 1; i < array.size(); i++)
            result = Math.max(result, array.get(i));

        return result;
    }

    public double sum()
    {
        /*
         * Uses Kahan summation algorithm, which is more accurate then
         * naively summing the values in floating point. Though it
         * does not guarenty the best possible accuracy
         *
         * See: http://en.wikipedia.org/wiki/Kahan_summation_algorithm
         */

        double sum = 0;
        double c = 0;
        for(double d : array)
        {
            double y = d - c;
            double t = sum+y;
            c = (t - sum) - y;
            sum = t;
        }

        return sum;
    }

    public double median()
    {
        ArrayList<Double> copy = new ArrayList<Double>(array);

        Collections.sort(copy);

        if(copy.size() % 2 == 1)
            return copy.get(copy.size()/2);
        else
            return copy.get(copy.size()/2)/2+copy.get(copy.size()/2)/2;//Divisions by 2 then add is more numericaly stable
    }

    public double mean()
    {
        return sum()/length();
    }

    public double standardDeviation()
    {
        double mu = mean();
        double tmp = 0;

        double N = length();


        for(double x : array)
            tmp += pow(x-mu, 2)/N;
        
        return sqrt(tmp);
    }

    public DenseVector sortedCopy()
    {
        ArrayList<Double> copy = new ArrayList<Double>(array);
        Collections.sort(copy);

        return new DenseVector(copy);
    }
    
}
