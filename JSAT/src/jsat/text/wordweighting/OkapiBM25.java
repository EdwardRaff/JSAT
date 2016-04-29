package jsat.text.wordweighting;

import java.util.List;
import jsat.linear.*;

/**
 * Implements the <a href="http://en.wikipedia.org/wiki/Okapi_BM25">Okapi BM25
 * </a> word weighting scheme. 
 * 
 * @author EdwardRaff
 */
public class OkapiBM25 extends WordWeighting
{

    private static final long serialVersionUID = 6456657674702490465L;
    private double k1;
    private double b;

    private double N;
    private double docAvg;
    /**
     * Okapi document frequency is the number of documents that contain a term, 
     * not the number of times it occurs
     */
    private int[] df;

    /**
     * Creates a new Okapi object
     */
    public OkapiBM25()
    {
        this(1.5, 0.75);
    }

    /**
     * Creates a new Okapi object
     * 
     * @param k1 the non negative coefficient to apply to the term frequency
     * @param b the coefficient to apply to the document length in the range [0,1]
     */
    public OkapiBM25(double k1, double b)
    {
        if(Double.isNaN(k1) || Double.isInfinite(k1) || k1 < 0)
            throw new IllegalArgumentException("coefficient k1 must be a non negative constant, not " + k1);
        this.k1 = k1;
        if(Double.isNaN(b) || b < 0 || b > 1)
            throw new IllegalArgumentException("coefficient b must be in the range [0,1], not " + b);
        this.b = b;
    }
    
    @Override
    public void setWeight(List<? extends Vec> allDocuments, List<Integer> df)
    {
        this.df = new int[df.size()];
        docAvg = 0;
        for( Vec v : allDocuments)
        {
            for(IndexValue iv : v)
            {
                docAvg += iv.getValue();
                this.df[iv.getIndex()]++;
            }
        }
        N = allDocuments.size();
        docAvg /= N;
        
    }

    @Override
    public void applyTo(Vec vec)
    {
        if(df == null)
            throw new RuntimeException("OkapiBM25 weightings haven't been initialized, setWeight method must be called before first use.");
        double sum = vec.sum();
        for(IndexValue iv : vec)
        {
            double value = iv.getValue();
            int index = iv.getIndex();
            double idf = Math.log( (N-df[index]+0.5)/(df[index]+0.5) );
        
        
            double result = idf * (value*(k1+1))/(value+k1*(1-b+b*sum/docAvg));
            vec.set(index, result);
        }
    }

    @Override
    public double indexFunc(double value, int index)
    {
        if (index < 0 || value == 0.0)
            return 0.0;
        
        return 0;
    }
    
}
