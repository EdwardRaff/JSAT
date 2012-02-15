
package jsat.math.optimization;

import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.math.Function;
import static java.lang.Math.*;
/**
 * The Rosenbrock function is a function with at least one minima with the value zero. It is often used as a benchmark for optimization problems. <br>
 * The minima is the vector of all ones. Once N %gt; 3, more then one minima can occur. 
 * 
 * @author Edward Raff
 */
public class RosenbrockFunction implements Function
{
    public double f(double... x)
    {
        return f(DenseVector.toDenseVec(x));
    }

    public double f(Vec x)
    {
        int N = x.length();
        double f = 0.0;
        for(int i = 0; i < N-1; i++)
        {
            double xi = x.get(i);
            f += pow(1.0-xi, 2)+100.0*pow(x.get(i+1)-xi*xi, 2);
        }
        
        return f;
    }
    
}
