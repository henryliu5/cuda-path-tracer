#ifndef __SPLIT_GEO_H__
#define __SPLIT_GEO_H__

#include "../scene/scene.h"

class SplitGeo
    : public MaterialSceneObject {
public:
	SplitGeo( Scene *scene, Material *mat, Geometry* geo )
		: MaterialSceneObject( scene, mat ),
          geo(geo)
        //   transform(geo->transform)
	{
        if(!geo){
            cout << "nullptr" << endl;
            exit(0);
        }
        setTransform(geo->transform);
	}
    
	virtual bool intersectLocal(ray& r, isect& i ) const { return geo->intersectLocal(r,i);}
	virtual bool hasBoundingBoxCapability() const { return true; }
    virtual BoundingBox ComputeLocalBoundingBox()
    {
        BoundingBox localbounds;
        return localbounds;
    }
protected:
    Geometry* geo;
	void glDrawLocal(int quality, bool actualMaterials, bool actualTextures) const {}
};
#endif // __SPLIT_GEO_H__
