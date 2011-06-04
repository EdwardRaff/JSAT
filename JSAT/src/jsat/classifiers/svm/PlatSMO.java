
package jsat.classifiers.svm;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPoint;
import jsat.distributions.kernels.KernelFunction;
import jsat.linear.Vec;
import static java.lang.Math.*;

/**
 *
 * @author Edward Raff
 */
public class PlatSMO extends SupportVectorMachine
{
    /**
     * Bias
     */
    protected double b = 0;
    private double C = 0.05;
    private double tolerance = 1e-3;
    private double epsilon = 1e-3;
    
    protected double[] alpha;
    protected double[] error;
    protected double bias;
    //Examine step may require a source of randomnes
    private Random rand;
    
    
    protected double[] label;

    public PlatSMO(KernelFunction kf)
    {
        super(kf, CacheMode.FULL);
        rand = new Random();
    }

    public CategoricalResults classify(DataPoint data)
    {
        if(vecs == null)
            throw new RuntimeException("Classifier has yet to be trained");
        
        double sum = 0;
        CategoricalResults cr = new CategoricalResults(2);
        
        for (int i = 0; i < vecs.length; i++)
            sum += alpha[i] * label[i] * kEval(vecs[i], data.getNumericalValues());


        //SVM only says yess / no, can not give a percentage
        if(sum > 0)
            cr.setProb(1, 1);
        else
            cr.setProb(0, 1);
        
        return cr;
    }

    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        trainC(dataSet);
    }

    public void trainC(ClassificationDataSet dataSet)
    {
        if(dataSet.getPredicting().getNumOfCategories() != 2)
            throw new ArithmeticException("SVM does not support non binary decisions");
        //First we need to set up the vectors array

        vecs = new Vec[dataSet.getSampleSize()];
        label = new double[vecs.length];
        for(int i = 0; i < vecs.length; i++)
        {
            DataPoint dataPoint = dataSet.getDataPoint(i);
            vecs[i] = dataPoint.getNumericalValues();
            if(dataSet.getDataPointCategory(i) == 0)
                label[i] = -1;
            else
                label[i] = 1;
        }
        
        setCacheMode(getCacheMode());//Initiates the cahce
        
        
        
        //initialize alpha array to all zero
        alpha = new double[vecs.length];//zero is default value
        error = new double[vecs.length];
        

        int numChanged = 0;
        boolean examinAll = true;

        while(examinAll || numChanged > 0)
        {
            numChanged = 0;
            if (examinAll)
            {
                //loop I over all training examples
                for (int i = 0; i < vecs.length; i++)
                    numChanged += examineExample(i);
            }
            else
            {
                //loop I over examples where alpha is not 0 & not C
                for (int i = 0; i < vecs.length; i++)
                    if (alpha[i] != 0 && alpha[i] != C)
                        numChanged += examineExample(i);
            }

            if(examinAll)
                examinAll = false;
            else if(numChanged == 0)
                examinAll = true;
        }

        //SVMs are usualy sparce, we dont need to keep all the original vectors!

        int supportVectorCount = 0;
        for(int i = 0; i < vecs.length; i++)
            if(alpha[i] > 0)//Its a support vector
            {
                vecs[supportVectorCount] = vecs[i];
                alpha[supportVectorCount] = alpha[i];
                label[supportVectorCount++] = label[i];
            }

        vecs = Arrays.copyOfRange(vecs, 0, supportVectorCount);
        alpha = Arrays.copyOfRange(alpha, 0, supportVectorCount);
        label = Arrays.copyOfRange(label, 0, supportVectorCount);
    }
    
    protected boolean takeStep(int i1, int i2)
    {
        if(i1 == i2)
            return false;
        //alph1 = Lagrange multiplier for i
        double alpha1 = alpha[i1], alpha2 = alpha[i2];
        //y1 = target[i1]
        double y1 = label[i1], y2 = label[i2];

        //E1 = SVM output on point[i1] - y1 (check in error cache)
        double E1, E2;
        if(alpha1 > 0  && alpha1 < C)
            E1 = error[i1];
        else
            E1 = decisionFunction(i1) - y1;
        
        if(alpha2 > 0  && alpha2 < C)
            E2 = error[i1];
        else
            E2 = decisionFunction(i2) - y2;

        //s = y1*y2
        double s = y1*y2;

        //Compute L, H : see smo-book, page 46
        double L, H;
        if(y1 != y2)
        {
            L = max(0, alpha2-alpha1);
            H = min(C, C+alpha2-alpha1);
        }
        else
        {
            L = max(0, alpha1+alpha2-C);
            H = min(C, alpha1+alpha2);
        }

        if (L == H)
            return false;

        double a1;//new alpha1
        double a2;//new alpha2

        /*
         * k11 = kernel(point[i1],point[i1])
         * k12 = kernel(point[i1],point[i2])
         * k22 = kernel(point[i2],point[i2]
         */
        double k11 = kEval(i1, i1);
        double k12 = kEval(i1, i2);
        double k22 = kEval(i2, i2);
        //eta = 2*k12-k11-k22
        double eta = 2*k12 - k11 - k22;

        if (eta < 0)
        {
            a2 = alpha2 - y2 * (E1 - E2) / eta;
            if (a2 < L)
                a2 = L;
            else if (a2 > H)
                a2 = H;
        }
        else
        {
            /*
             * Lobj = objective function at a2=L
             * Hobj = objective function at a2=H
             */
            double c1 = eta / 2;
            double c2 = y2 * (E1 - E2) - eta * alpha2;
            double Lobj = c1 * L * L + c2 * L;
            double Hobj = c1 * H * H + c2 * H;

            if(Lobj > Hobj + epsilon)
                a2 = L;
            else if(Lobj < Hobj - epsilon)
                a2 = H;
            else
                a2 = alpha2;
        }

        if(a2 < 1e-8)
            a2 = 0;
        else if (a2 > C - 1e-8)
            a2 = C;

        if(abs(a2 - alpha2) < epsilon*(a2+alpha2+epsilon))
            return false;

        a1 = alpha1 + s *(alpha2-a2);

        if (a1 < 0)
        {
            a2 += s * a1;
            a1 = 0;
        }
        else if (a1 > C)
        {
            double t = a1 - C;
            a2 += s * t;
            a1 = C;
        }

        

        double oldB = b;
        
        if(a1 > 0 && a1 < C)
        {
            b = E1 + y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12 + b;
        }
        else if(a2 > 0 && a2 < C)
        {
            b = E2 + y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22 + b;
        }
        else
        {
            //Update threshold to reflect change in Lagrange multipliers : see smo-book, page 49
            double b1 = E1 + y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12 + b;
            double b2 = E2 + y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22 + b;
            
            b = (b1+b2)/2;
        }

        //Update error cache using new Lagrange multipliers: smo-book, see page 49 (12.11)

        //Pre compute
        double y1ADelta = y1 * (a1 - alpha1);
        double y2ADelta = y2 * (a2 - alpha2);

        for (int i = 0; i < vecs.length; i++)
        {
            if (0 < alpha[i] && alpha[i] < C)
                error[i] += y1ADelta * kEval(i1, i) + y2ADelta * kEval(i2,i) - (b-oldB);
        }
        
        error[i1] = error[i2] = 0;

        //Store a1 in the alpha array
        alpha[i1] = a1;
        //Store a2 in the alpha arra
        alpha[i2] = a2;


        return true;
    }
    
    private int examineExample(int i2)
    {
        //y2 = target[i2]
        double y2 = label[i2];
        //alph2 = Lagrange multiplier for i2
        double alpha2 = alpha[i2];

        //E2 = SVM output on point[i2] - y2 (check in error cache)
        double E2;
        if(alpha2 > 0  && alpha2 < C)
            E2 = error[i2];
        else
            E2 = decisionFunction(i2) - y2;

        double r2 = E2*y2;

        if(!((r2 < -tolerance && alpha2 < C) || (r2 > tolerance && alpha2 > 0)))
            return 0;

        //Second Choice Heuristic: smo-book, page 78
        int i1 = -1;
        double maxError = 0;//minimized on the largest error
        for (int i = 0; i < vecs.length; i++)
        {
            if (alpha[i] > 0 && alpha[i] < C)
            {//This method is only describe for when the cache has a value, if ther eis no value in the cache we will not use it
                double aux = abs(E2 - error[i]);
                if (aux > maxError)
                {
                    maxError = aux;
                    i1 = i;
                }
            }
        }

        if (i1 >= 0 && takeStep(i1, i2))
            return 1;


        //next Heuristic:
        //loop over all non-zero and non-C alpha, starting at random point
        int randomIndex = rand.nextInt(vecs.length);
        for (i1 = randomIndex; i1 < vecs.length; i1++)
            if (alpha[i1] > 0 && alpha[i1] < C)
                if (takeStep(i1, i2))
                    return 1;
        //Start back at the front
        for (i1 = 0; i1 < randomIndex; i1++)
            if (alpha[i1] > 0 && alpha[i1] < C)
                if (takeStep(i1, i2))
                    return 1;

        //oh noes! Nothign worked, do the same thing but check ALL messages
        randomIndex = rand.nextInt(vecs.length);
        for (i1 = randomIndex; i1 < vecs.length; i1++)
            if (takeStep(i1, i2))
                return 1;
        //Start back at the front
        for (i1 = 0; i1 < randomIndex; i1++)
            if (takeStep(i1, i2))
                return 1;



        return 0;
    }
    
    protected double decisionFunction(int v)
    {
        double sum = 0;
        for(int i = 0; i < vecs.length; i++)
            if(alpha[i] > 0)
                sum += alpha[i] * label[i] * kEval(i, v);

        return sum-b;
    }

    
    
    
}
