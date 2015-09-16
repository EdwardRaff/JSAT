package jsat.lossfunctions;

import jsat.classifiers.CategoricalResults;
import jsat.linear.Vec;

/**
 * The HingeLoss loss function for classification <i>L(x, y) = max(0, 1-y*x)</i>
 * . This also includes the multi-class version of the hinge loss. <br>
 * This function is only once differentiable.
 *
 * @author Edward Raff
 */
public class HingeLoss implements LossMC {

  private static final long serialVersionUID = -7001702646530236153L;

  public static CategoricalResults classify(final double score) {
    final CategoricalResults cr = new CategoricalResults(2);
    if (score > 0) {
      cr.setProb(1, 1.0);
    } else {
      cr.setProb(0, 1.0);
    }
    return cr;
  }

  /**
   * Computes the first derivative of the HingeLoss loss
   *
   * @param pred
   *          the predicted value
   * @param y
   *          the target value
   * @return the first derivative of the HingeLoss loss
   */
  public static double deriv(final double pred, final double y) {
    if (pred * y > 1) {
      return 0;
    } else {
      return -y;
    }
  }

  /**
   * Computes the HingeLoss loss
   *
   * @param pred
   *          the predicted value
   * @param y
   *          the target value
   * @return the HingeLoss loss
   */
  public static double loss(final double pred, final double y) {
    return Math.max(0, 1 - y * pred);
  }

  @Override
  public HingeLoss clone() {
    return this;
  }

  @Override
  public void deriv(final Vec processed, final Vec derivs, final int y) {
    final double proccessed_y = processed.get(y);
    double maxVal_not_y = Double.NEGATIVE_INFINITY;
    int maxIndx = -1;
    for (int i = 0; i < processed.length(); i++) {
      if (i != y && processed.get(i) > maxVal_not_y) {
        maxIndx = i;
        maxVal_not_y = processed.get(i);
      }
    }

    derivs.zeroOut();
    if (1.0 + maxVal_not_y - proccessed_y > 0) {
      derivs.set(y, -1.0);
      derivs.set(maxIndx, 1.0);
    }
  }

  @Override
  public CategoricalResults getClassification(final double score) {
    return classify(score);
  }

  @Override
  public CategoricalResults getClassification(final Vec processed) {
    int maxIndx = 0;
    double maxVal_not_y = processed.get(maxIndx);
    for (int i = 1; i < processed.length(); i++) {
      if (processed.get(i) > maxVal_not_y) {
        maxIndx = i;
        maxVal_not_y = processed.get(i);
      }
    }
    final CategoricalResults toRet = new CategoricalResults(processed.length());
    toRet.setProb(maxIndx, 1.0);
    return toRet;
  }

  @Override
  public double getDeriv(final double pred, final double y) {
    return deriv(pred, y);
  }

  @Override
  public double getDeriv2(final double pred, final double y) {
    return 0;
  }

  @Override
  public double getDeriv2Max() {
    return 0;
  }

  @Override
  public double getLoss(final double pred, final double y) {
    return loss(pred, y);
  }

  @Override
  public double getLoss(final Vec processed, final int y) {
    double max_not_y = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < processed.length(); i++) {
      if (i != y) {
        max_not_y = Math.max(max_not_y, processed.get(i));
      }
    }
    return Math.max(0, 1.0 + max_not_y - processed.get(y));
  }

  @Override
  public void process(final Vec pred, final Vec processed) {
    if (pred != processed) {
      pred.copyTo(processed);
    }
  }
}
