
package jsat.linear;

import static java.lang.Math.pow;
import static java.lang.Math.sqrt;
import java.util.*;
import jsat.math.Function;
import jsat.math.IndexFunction;

/**
 * Provides a vector implementation that is sparse. It does not allocate space 
 * for a vector of the specified size, and only stores non zero values. All 
 * values not stored are implicitly zero. 
 * <br>
 * Operations that change several zero values in a sparse vector to non-zero 
 * values may have degraded performance. 
 * <br>
 * Sparce vector should never be used unless at least half the values are zero. 
 * If more then half the values are non-zero, it will use more memory then an 
 * equivalent {@link DenseVector}. The more values that are zero in the vector, 
 * the better its performance will be. 
 * 
 * @author Edward Raff
 */
public class SparseVector extends  Vec
{
    /**
     * Length of the vector
     */
    private int length;
    /**
     * number of indices used in this vector
     */
    protected int used;
    /**
     * The mapping to true index values
     */
    protected int[] indexes;
    /**
     * The Corresponding values for each index
     */
    protected double[] values;
    
    private Double sumCache = null;
    private Double varianceCache = null;
    private Double minCache = null;
    private Double maxCache = null;
    
    public SparseVector(int length)
    {
        this(length, 10);
    }

    public SparseVector(List<Double> vals)
    {
        this(vals.size());
        int z = 0;
        for(int i = 0; i < vals.size(); i++)
            if(vals.get(i) != 0)
            {
                if(z >= indexes.length)
                {
                    indexes = Arrays.copyOf(indexes, indexes.length*3/2);
                    values = Arrays.copyOf(values, values.length*3/2);
                }
                indexes[z] = i;
                values[z++] = vals.get(i);
            }
    }
    
    public SparseVector(int length, int capacity)
    {
        if(length <= 0)
            throw new ArithmeticException("Vector must have a positive dimension");
        this.used = 0;
        capacity = Math.max(capacity, 10);
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
    
    @Override
    public int length()
    {
        return length;
    }

    /**
     * Because sparce vectors do not have most value set, they can 
     * have their length increased, and sometimes decreased, without 
     * any effort. The length can always be extended. The length can
     * be reduced down to the size of the largest non zero element. 
     * 
     * @param length the new length of this vector
     */
    public void setLength(int length)
    {
        if(used > 0 && length < indexes[used-1])
            throw new RuntimeException("Can not set the length to a value less then an index already in use");
        this.length = length;
    }

    /**
     * @return the number of non zero elements in the vector.
     */
    public int nnz()
    {
        return used;
    }
    
    /**
     * Increments the value at the given index by the given value. 
     * @param index the index of the value to alter
     * @param val the value to be added to the index
     */
    @Override
    public void increment(int index, double val)
    {
        if (index > length - 1 || index < 0)
            throw new ArithmeticException("Can not access an index larger then the vector or a negative index");
        
        int location = Arrays.binarySearch(indexes, 0, used, index);
        if(location < 0)
            insertValue(location, index, val);
        else
            values[location]+=val;
    }
    
    @Override
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

    @Override
    public void set(int index, double val)
    {
        if(index > length()-1 || index < 0)
            throw new ArithmeticException("Can not set an index larger then the array");

        
        clearCaches();
        int insertLocation = Arrays.binarySearch(indexes, 0, used, index);
        if(insertLocation >= 0)
            values[insertLocation] = val;
        else
            insertValue(insertLocation, index, val);
    }

    /**
     * Takes the negative insert location value returned by {@link Arrays#binarySearch(int[], int, int, int) } 
     * and adjust the vector to add the given value into this location. Should only be called with negative 
     * input returned by said method. Should never be called for an index that in fact does already exist 
     * in this sparce vector. 
     * 
     * @param insertLocation the negative insertion index such that -(insertLocation+1) is the address that the value should have
     * @param index the index that is being added
     * @param val the value that is being added for the given index
     */
    private void insertValue(int insertLocation, int index, double val)
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

    @Override
    public Vec add(Vec v)
    {
        if(this.length() != v.length())
            throw new ArithmeticException("Vectors must have the same length");

        
        if(v instanceof SparseVector)
        {
            SparseVector ret = new SparseVector(length());
            SparseVector b = (SparseVector) v;
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

    @Override
    public Vec subtract(Vec v)
    {
        if(this.length() != v.length())
            throw new ArithmeticException("Vectors must have the same length");

        
        if(v instanceof SparseVector)
        {
            SparseVector ret = new SparseVector(length());
            SparseVector b = (SparseVector) v;
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


    @Override
    public Vec sortedCopy()
    {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public double min()
    {
        if(minCache != null)
            return minCache;
        double result = 0;
        for(int i = 0; i < used; i++)
            result = Math.min(result, values[i]);

        return (minCache = result);
    }

    @Override
    public double max()
    {
        if(maxCache != null)
            return maxCache;
        
        double result = 0;
        for(int i = 0; i < used; i++)
            result = Math.max(result, values[i]);

        return (maxCache = result);
    }

    @Override
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

    @Override
    public double mean()
    {
        return sum()/length();
    }

    @Override
    public double standardDeviation()
    {
        return Math.sqrt(variance());
    }

    @Override
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

    @Override
    public double median()
    {
        throw new UnsupportedOperationException("Not supported yet.");
    }
    
    @Override
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

    @Override
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

    @Override
    public double dot(Vec v)
    {
        double dot = 0;
        
        if(v instanceof SparseVector)
        {
            SparseVector b = (SparseVector) v;
            int p1 = 0, p2 = 0;
            while (p1 < used && p2 < b.used)
            {
                int a1 = indexes[p1], a2 = b.indexes[p2];
                if (a1 == a2)
                    dot += values[p1++] * b.values[p2++];
                else if (a1 > a2)
                    p2++;
                else
                    p1++;
            }
        }
        else// it is dense
            for (int i = 0; i < used; i++)
                dot += values[i] * v.get(indexes[i]);

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
                sb.append("0.0");
        }
        sb.append("]");
        
        return sb.toString();
    }

    @Override
    public Vec multiply(double c)
    {
        SparseVector sv = new SparseVector(length, used);
        
        for(int i = 0; i < used; i++)
            sv.values[i] *= c;
        
        return sv;
    }

    @Override
    public void multiply(Matrix A, Vec b)
    {
        if(this.length() != A.rows())
            throw new ArithmeticException("Vector x Matrix dimensions do not agree");
        else if(b.length() != A.cols())
            throw new ArithmeticException("Destination vector is not the right size");
        
        for(int i = 0; i < used; i++)
        {
            double val = this.values[i];
            int index = this.indexes[i];
            for(int j = 0; j < A.cols(); j++)
                b.increment(j, val*A.get(index, j));
        }
    }
    
    @Override
    public Vec divide(double c)
    {
        SparseVector sv = new SparseVector(length, used);
        
        for(int i = 0; i < used; i++)
            sv.values[i] /= c;
        
        return sv;
    }

    @Override
    public Vec add(double c)
    {
        SparseVector sv = new SparseVector(length, used);
        
        for(int i = 0; i < used; i++)
            sv.values[i] += c;
        
        return sv;
    }

    @Override
    public void mutableAdd(double c)
    {
        clearCaches();
        /* This NOT the most efficient way to implement this. 
         * But adding a constant to every value in a sparce 
         * vector defeats its purpos. 
         */
        for(int i = 0; i < length(); i++)
            this.set(i, get(i) + c);
    }

    @Override
    public void mutableAdd(double c, Vec v)
    {
        clearCaches();
        if(v instanceof SparseVector)
        {
            SparseVector b = (SparseVector) v;
            int p1 = 0, p2 = 0;
            while (p1 < used && p2 < b.used)
            {
                int a1 = indexes[p1], a2 = b.indexes[p2];
                if (a1 == a2)
                {
                    values[p1] += c*b.values[p2];
                    p1++;
                    p2++;
                }
                else if (a1 > a2)
                {
                    //0 + some value is that value, set it 
                    this.set(a2, c*b.values[p2]);
                    /*
                     * p2 must be increment becase were moving to the next value
                     * 
                     * p1 must be be incremented becase a2 was less thenn the current index. 
                     * So the inseration occured before p1, so for indexes[p1] to == a1, 
                     * p1 must be incremented
                     * 
                     */
                    p1++;
                    p2++;
                }
                else//a1 < a2, thats adding 0 to this vector, nothing to do. 
                {
                    p1++;
                }
            }
            
            //One of them is now empty. 
            //If b is not empty, we must add b to this. If b is empty, we would be adding zeros to this [so we do nothing]
            while(p2 < b.used)
                this.set(b.indexes[p2], c*b.values[p2++]);//TODO Can be done more efficently 
        }
        else
        {
            //Else it is dense
            for(int i = 0; i < length(); i++)
                this.set(i, this.get(i) + c*v.get(i));
        }
        
    }

    @Override
    public void mutableMultiply(double c)
    {
        clearCaches();
        
        for(int i = 0; i < used; i++)//0*c = 0, so we can do this sparcly
            values[i] *= c;
    }

    @Override
    public void mutableDivide(double c)
    {
        clearCaches();
        
        for(int i = 0; i < used; i++)//0/c = 0, so we can do this sparcly
            values[i] /= c;
    }

    @Override
    public double pNormDist(double p, Vec y)
    {
        if(this.length() != y.length())
            throw new ArithmeticException("Vectors must be of the same length");
        
        double norm = 0;
        
        if (y instanceof SparseVector)
        {
            int p1 = 0, p2 = 0;
            SparseVector b = (SparseVector) y;            
            
            while (p1 < this.used && p2 < b.used)
            {
                int a1 = indexes[p1], a2 = b.indexes[p2];
                if (a1 == a2)
                {
                    norm += Math.pow(Math.abs(this.values[p1] - b.values[p2]), p);
                    p1++;
                    p2++;
                }
                else if (a1 > a2)
                    norm += Math.pow(Math.abs(b.values[p2++]), p);
                else//a1 < a2, this vec has a value, other does not
                    norm += Math.pow(Math.abs(this.values[p1++]), p);
            }
            //One of them is now empty. 
            //So just sum up the rest of the elements
            while(p1 < this.used)
                norm += Math.pow(Math.abs(this.values[p1++]), p);
            while(p2 < b.used)
                norm += Math.pow(Math.abs(b.values[p2++]), p);
        }
        else
        {
            int z = 0;
            for (int i = 0; i < length(); i++)
            {
                //Move through until we hit the next null element, comparing the other vec to zero
                while (z < used && indexes[z] > i)
                    norm += Math.pow(Math.abs(-y.get(i)), p);

                //We made it! (or are at the end). Is our non zero value the same?
                if (z < used && indexes[z] == i)
                    norm += Math.pow(Math.abs(values[z] - y.get(i)), p);
            }
        }
        return Math.pow(norm, 1.0/p);
    }

    @Override
    public double pNorm(double p)
    {
        double norm = 0;
        
        for(int i = 0; i < used; i++)
            norm += Math.pow(Math.abs(values[i]), p);
        
        return Math.pow(norm, 1.0/p);
    }
    
    @Override
    public Vec clone()
    {
        SparseVector copy = new SparseVector(length, Math.max(used, 10));
        
        System.arraycopy(this.values, 0, copy.values, 0, this.used);
        System.arraycopy(this.indexes, 0, copy.indexes, 0, this.used);
        copy.used = this.used;
        
        return copy;
    }

    @Override
    public Vec normalized()
    {
        Vec copy = this.clone();
        copy.normalize();
        return copy;
    }

    @Override
    public void normalize()
    {
        double sum = 0;

        for(int i = 0; i < used; i++)
            sum += values[i]*values[i];
        
        sum = Math.sqrt(sum);

        mutableDivide(sum); 
    }

    @Override
    public Vec pairwiseMultiply(Vec b)
    {
        if(this.length() != b.length())
            throw new ArithmeticException("Vectors must have the same length");
        SparseVector toRet = (SparseVector) this.clone();
        
        toRet.mutablePairwiseMultiply(b);
        
        return toRet;
    }

    @Override
    public Vec pairwiseDivide(Vec b)
    {
        if(this.length() != b.length())
            throw new ArithmeticException("Vectors must have the same length");
        SparseVector toRet = (SparseVector) this.clone();
        
        toRet.mutablePairwiseDivide(b);
        
        return toRet;
    }

    @Override
    public void mutablePairwiseMultiply(Vec b)
    {
        if(this.length() != b.length())
            throw new ArithmeticException("Vectors must have the same length");
        clearCaches();
        
        //TODO a space , sparce optimized version could be done
        for(int i = 0; i <= used; i++)
            values[i] *= b.get(indexes[i]);//zeros stay zero
    }

    @Override
    public void mutablePairwiseDivide(Vec b)
    {
        if(this.length() != b.length())
            throw new ArithmeticException("Vectors must have the same length");
        clearCaches();
        
        //TODO a space , sparce optimized version could be done
        for(int i = 0; i <= used; i++)
            values[i] /= b.get(indexes[i]);//zeros stay zero
    }
    
    @Override
    public boolean equals(Object obj)
    {
        if(!(obj instanceof Vec))
            return false;
        Vec otherVec = (Vec) obj;
        
        if(this.length() != otherVec.length())
            return false;
        
        
            int z = 0;
            for(int i = 0; i < length(); i++)
            {
                //Move through until we hit the next null element, comparing the other vec to zero
                while(z < used && indexes[z] > i)
                    if(otherVec.get(i++) != 0)
                        return false;
                
                //We made it! (or are at the end). Is our non zero value the same?
                if(z < used && indexes[z] == i)
                    if(values[z++] != otherVec.get(i))
                        return false;
            }
        
        
        return true;
    }

    @Override
    public boolean equals(Object obj, double range)
    {
        if(!(obj instanceof Vec))
            return false;
        Vec otherVec = (Vec) obj;
        range = Math.abs(range);
        
        if(this.length() != otherVec.length())
            return false;
        

        int z = 0;
        for (int i = 0; i < length(); i++)
        {
            //Move through until we hit the next null element, comparing the other vec to zero
            while (z < used && indexes[z] > i)
                if (Math.abs(otherVec.get(i++)) > range)//We are zero!
                    return false;

            //We made it! (or are at the end). Is our non zero value the same?
            if (z < used && indexes[z] == i)
                if (Math.abs(values[z++] - otherVec.get(i)) > range)
                    return false;
        }


        return true;
    }

    @Override
    public double[] arrayCopy()
    {
        double[] array = new double[length()];
        
        for(int i = 0; i < used; i++)
            array[indexes[i]] = values[i];
        
        return array;
    }

    @Override
    public void applyFunction(Function f)
    {
        if(f.f(0.0) != 0.0)
            super.applyFunction(f);
        else//Then we only need to apply it to the non zero values! 
        {
            for(int i = 0; i < used; i++)
                values[i] = f.f(values[i]);
        }
    }

    @Override
    public void applyIndexFunction(IndexFunction f)
    {
        if(f.f(0.0, -1) != 0.0)
            super.applyIndexFunction(f);
        else//Then we only need to apply it to the non zero values! 
        {
            for(int i = 0; i < used; i++)
                values[i] = f.indexFunc(values[i], i);
        }
    }

    @Override
    public void zeroOut()
    {
        this.used = 0;
    }

    @Override
    public Iterator<IndexValue> getNonZeroIterator()
    {
        Iterator<IndexValue> itor = new Iterator<IndexValue>() 
        {
            int curUsedPos = 0;
            IndexValue indexValue = new IndexValue(-1, Double.NaN);
            public boolean hasNext()
            {
                return curUsedPos < used;
            }

            public IndexValue next()
            {
                indexValue.setIndex(indexes[curUsedPos]);
                indexValue.setValue(values[curUsedPos++]);
                return indexValue;
            }

            public void remove()
            {
                throw new UnsupportedOperationException("Not supported yet.");
            }
        };
        return itor;
    }
    
    @Override
    public int hashCode()
    {
        int result = 1;
        
        for (int i = 0; i < used; i++) 
        {
            long bits = Double.doubleToLongBits(values[i]);
            result = 31 * result + (int)(bits ^ (bits >>> 32));
            result = 31 * result + indexes[i];
        }
        
        return 31* result + length;
    }

    @Override
    public boolean isSparse()
    {
        return true;
    }
}
