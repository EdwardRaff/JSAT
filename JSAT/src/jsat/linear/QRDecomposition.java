
package jsat.linear;

import java.io.Serializable;
import java.util.concurrent.ExecutorService;
import static java.lang.Math.*;

/**
 *
 * @author Edward Raff
 */
public class QRDecomposition implements Serializable
{

	private static final long serialVersionUID = 7578073062361216223L;
	private Matrix Q_T, R;

    public QRDecomposition(Matrix Q, Matrix R)
    {
        if(!Q.isSquare())
            throw new ArithmeticException("Q is always square, rectangular Q is invalid");
        else if(Q.rows() != R.rows())
            throw new ArithmeticException("Q and R do not agree");
        
        this.Q_T = Q;
        this.Q_T.mutableTranspose();
        this.R = R;
    }
    
    public QRDecomposition(Matrix A)
    {
        Matrix[] qr = A.clone().qr();
        Q_T = qr[0];
        Q_T.mutableTranspose();
        R = qr[1];
    }
    
    public QRDecomposition(Matrix A, ExecutorService threadpool)
    {
        Matrix[] qr = A.clone().qr(threadpool);
        Q_T = qr[0];
        Q_T.mutableTranspose();
        R = qr[1];
    }
    
    /**
     * 
     * @return the absolute value of the determinant of the original Matrix, abs(|A|)
     */
    public double absDet()
    {
        if(!R.isSquare())
            throw new ArithmeticException("Can only compute the determinant of a square matrix");
            
        double absD = 1;
        for(int i = 0; i < min(R.rows(), R.cols()); i++)
            absD *= R.get(i, i);
        
        return abs(absD);
    }
    
    public Vec solve(Vec b)
    {
        if(b.length() != R.rows())
            throw new ArithmeticException("Matrix vector dimensions do not agree");
        //A * x = b, we want x
        //QR x = b
        //R * x = Q' * b
        
        Vec y = Q_T.multiply(b);
        
        //Solve R * x = y using back substitution
        Vec x = LUPDecomposition.backSub(R, y);
        
        return x;
    }
    
    public Matrix solve(Matrix B)
    {
        //A * x = B, we want x
        //QR x = b
        //R * x = Q' * b
        
        Matrix y = Q_T.multiply(B);
        
        //Solve R * x = y using back substitution
        Matrix x = LUPDecomposition.backSub(R, y);
        
        return x;
    }
    
    public Matrix solve(Matrix B, ExecutorService threadpool)
    {
        //A * x = B, we want x
        //QR x = b
        //R * x = Q' * b
        
        Matrix y = Q_T.multiply(B, threadpool);
        
        //Solve R * x = y using back substitution
        Matrix x = LUPDecomposition.backSub(R, y, threadpool);
        
        return x;
    }
}
