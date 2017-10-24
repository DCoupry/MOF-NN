subroutine StructureFactors(positions,numbers,Fhkl)
    real   , intent(in )  :: positions(:,:)
    integer, intent(in )  :: numbers(:)
    real   , intent(out)  :: Fhkl(31,31,31,2)
    integer :: h,k,l
    complex :: f_hkl
    real   , parameter :: pi = 3.1415927
    complex, parameter :: j  = cmplx(0,1)
    Fhkl = 0.0
    nat = size(positions,dim=1)
    do h=-15,15
        do k=-15,15
            do l=-15,15
                f_hkl = dot_product(exp(2.0*j*pi*(h*positions(:,1)+k*positions(:,2)+l*positions(:,3))),numbers)
                Fhkl(h+16,k+16,l+16,1) = abs(f_hkl)
                Fhkl(h+16,k+16,l+16,2) = atan2(aimag(f_hkl),real(f_hkl))
            end do
        end do
    end do
end subroutine

