//
// $Id: BoxDomain.cpp,v 1.7 2001-07-17 23:02:19 lijewski Exp $
//

#include <BoxDomain.H>

#ifdef BL_NAMESPACE
namespace BL_NAMESPACE
{
#endif

const Box&
BoxDomain::operator[] (const BoxDomainIterator& bli) const
{
    return lbox[bli];
}

bool
BoxDomain::contains (const IntVect& v) const
{
    return BoxList::contains(v);
}

bool
BoxDomain::contains (const Box& b) const
{
    return BoxList::contains(b);
}

bool
BoxDomain::contains (const BoxList& bl) const
{
    return BoxList::contains(bl);
}

BoxDomain&
BoxDomain::intersect (const Box& b)
{
    BoxList::intersect(b);
    BL_ASSERT(ok());
    return *this;
}

void
intersect (BoxDomain&       dest,
           const BoxDomain& fin,
           const Box&       b)
{
   dest = fin;
   dest.intersect(b);
}

BoxDomain&
BoxDomain::refine (int ratio)
{
    BoxList::refine(ratio);
    BL_ASSERT(ok());
    return *this;
}

void
refine (BoxDomain&       dest,
        const BoxDomain& fin,
        int              ratio)
{
    dest = fin;
    dest.refine(ratio);
}

void
accrete (BoxDomain&       dest,
         const BoxDomain& fin,
         int              sz)
{
    dest = fin;
    dest.accrete(sz);
}

void
coarsen (BoxDomain&       dest,
         const BoxDomain& fin,
         int              ratio)
{
    dest = fin;
    dest.coarsen(ratio);
}

BoxDomain&
BoxDomain::complementIn (const Box&       b,
                         const BoxDomain& bl)
{
    BoxList::complementIn(b,bl);
    BL_ASSERT(ok());
    return *this;
}

BoxDomain
complementIn (const Box&       b,
              const BoxDomain& bl)
{
    BoxDomain result;
    result.complementIn(b,bl);
    return result;
}

BoxDomain&
BoxDomain::shift (int dir,
                  int nzones)
{
    BoxList::shift(dir, nzones);
    return *this;
}

BoxDomain&
BoxDomain::shiftHalf (int dir,
                      int num_halfs)
{
    BoxList::shiftHalf(dir, num_halfs);
    return *this;
}

BoxDomain&
BoxDomain::shiftHalf (const IntVect& iv)
{
    BoxList::shiftHalf(iv);
    return *this;
}

BoxList
BoxDomain::boxList () const
{
    return BoxList(*this);
}

bool
BoxDomain::operator== (const BoxDomain& rhs) const
{
    return BoxList::operator==(rhs);
}

bool
BoxDomain::operator!= (const BoxDomain& rhs) const
{
    return !BoxList::operator==(rhs);
}

BoxDomain::BoxDomain ()
    : BoxList(IndexType::TheCellType())
{}

BoxDomain::BoxDomain (IndexType _ctype)
    : BoxList(_ctype)
{}

BoxDomain::BoxDomain (const BoxDomain& rhs)
    : BoxList(rhs)
{}

BoxDomain&
BoxDomain::operator= (const BoxDomain& rhs)
{
    BoxList::operator=(rhs);
    return *this;
}

BoxDomain::~BoxDomain ()
{}

void
BoxDomain::add (const Box& b)
{
    BL_ASSERT(b.ixType() == ixType());

    List<Box> check;
    check.append(b);
    for (ListIterator<Box> bli(lbox); bli; ++bli)
    {
        List<Box> tmp;
        for (ListIterator<Box> ci(check); ci; )
        {
            if (ci().intersects(bli()))
            {
                //
                // Remove c from the check list, compute the
                // part of it that is outside bln and collect
                // those boxes in the tmp list.
                //
                BoxList tmpbl(boxDiff(ci(), bli()));
                tmp.catenate(tmpbl.listBox());
                check.remove(ci);
            }
            else
                ++ci;
        }
        check.catenate(tmp);
    }
    //
    // At this point, the only thing left in the check list
    // are boxes that nowhere intersect boxes in the domain.
    //
    lbox.catenate(check);
    BL_ASSERT(ok());
}

void
BoxDomain::add (const BoxList& bl)
{
    for (BoxListIterator bli(bl); bli; ++bli)
        add(*bli);
}

BoxDomain&
BoxDomain::rmBox (const Box& b)
{
    BL_ASSERT(b.ixType() == ixType());

    List<Box> tmp;

    for (ListIterator<Box> bli(lbox); bli; )
    {
        if (bli().intersects(b))
        {
            BoxList tmpbl(boxDiff(bli(),b));
            tmp.catenate(tmpbl.listBox());
            lbox.remove(bli);
        }
        else
            ++bli;
    }
    lbox.catenate(tmp);
    return *this;
}

bool
BoxDomain::ok () const
{
    //
    // First check to see if boxes are valid.
    //
    bool status = BoxList::ok();
    if (status)
    {
        //
        // Now check to see that boxes are disjoint.
        //
        for (BoxListIterator bli(*this); bli; ++bli)
        {
            BoxListIterator blii(bli); ++blii;
            while (blii)
            {
                if (bli().intersects(blii()))
                {
                    std::cout << "Invalid DOMAIN, boxes overlap" << '\n'
                              << "b1 = " << bli() << '\n'
                              << "b2 = " << blii() << '\n';
                    status = false;
                }
                ++blii;
            }
        }
    }
    return status;
}

BoxDomain&
BoxDomain::accrete (int sz)
{
    BoxList bl(*this);
    bl.accrete(sz);
    clear();
    add(bl);
    return *this;
}

BoxDomain&
BoxDomain::coarsen (int ratio)
{
    BoxList bl(*this);
    bl.coarsen(ratio);
    clear();
    add(bl);
    return *this;
}

std::ostream&
operator<< (std::ostream&    os,
            const BoxDomain& bd)
{
    os << "(BoxDomain " << BoxList(bd) << ")" << std::flush;
    if (os.fail())
        BoxLib::Error("operator<<(ostream&,BoxDomain&) failed");
    return os;
}

#ifdef BL_NAMESPACE
}
#endif

